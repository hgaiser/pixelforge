use super::{CTB_SIZE, H265, level};
use crate::encoder::references::ReferenceSet;

use crate::converter::RgbFormat;
use crate::encoder::EncodeConfig;
use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, build_encoder_common, check_quality_level, query_video_caps,
};
use crate::encoder::profile::EncodeProfile;
use crate::encoder::profile::h265_profile_idc;
use crate::encoder::resources::MIN_BITSTREAM_BUFFER_SIZE;
use crate::encoder::rgb::RgbConversionCaps;
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

impl H265 {
    /// Create a new H.265/HEVC encoder.
    pub(crate) fn create(
        context: VideoContext,
        config: EncodeConfig,
        rgb_input: Option<RgbFormat>,
    ) -> Result<CodecEncoder<Self>> {
        info!(
            "Creating H.265 encoder: {}x{}, chroma_format={:?}",
            config.dimensions.width, config.dimensions.height, config.chroma_format
        );

        let profile_idc =
            h265_profile_idc(config.chroma_format, config.bit_depth).ok_or_else(|| {
                PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format / bit depth for H.265: {:?} / {:?}",
                    config.chroma_format, config.bit_depth
                ))
            })?;

        let profile = EncodeProfile::new(&config, rgb_input.is_some())?;
        let profile_info = profile.info();

        // Query device capabilities with the H.265-specific capability struct
        // chained in (required by the driver).
        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut h265_caps);
        let mut rgb_caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        if rgb_input.is_some() {
            capabilities = capabilities.push(&mut rgb_caps);
        }
        let caps = query_video_caps(&context, profile_info, &mut capabilities)?;
        check_quality_level(&config, encode_caps.max_quality_levels)?;
        let rgb_caps = rgb_input.map(|_| RgbConversionCaps::from(&rgb_caps));

        // Scale the bitstream buffer by resolution so a single frame can't overflow (matches AV1).
        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        // The DPB holds what the level allows at the stream's size, so a
        // window of references cannot make the stream break its level.
        let rate = config.frame_rate;
        let frame_rate = rate.numerator.div_ceil(rate.denominator.max(1));
        let max_level_idc = h265_caps.max_level_idc;
        let max_dpb_pictures = |width: u32, height: u32| {
            let level = level::level_for(
                &level::Needs {
                    width,
                    height,
                    frame_rate,
                },
                max_level_idc,
            );
            level::max_dpb_size(width, height, level) as usize
        };

        let init = build_encoder_common(CommonInitRequest {
            context: &context,
            config: &config,
            profile,
            rgb_input,
            caps: &caps,
            align_unit: CTB_SIZE,
            max_reference_window: 15,
            max_dpb_pictures: Some(&max_dpb_pictures),
            max_frame_references: h265_caps.max_p_picture_l0_reference_count as usize,
            bitstream_buffer_size,
            allow_layered_dpb: true,
            rgb_caps,
            b_frame_support: crate::encoder::bframes::h265_support(
                context.device_properties().vendor_id,
                &h265_caps,
                caps.max_active_reference_pictures,
            ),
        })?;
        let reference_window = init.reference_window;
        let active_reference_count = init.active_reference_count;
        let b_l0_reference_count = (init.session_active_references.saturating_sub(1))
            .min(h265_caps.max_b_picture_l0_reference_count);
        let common = init.common;
        let level_idc = level::level_for(
            &level::Needs {
                width: common.aligned_width,
                height: common.aligned_height,
                frame_rate,
            },
            max_level_idc,
        );
        info!(
            "H.265 level {level_idc} (as StdVideoH265LevelIdc), DPB of {} pictures",
            common.dpb_slot_count
        );

        let codec = H265 {
            header_data: None,
            references: ReferenceSet::new(reference_window),
            active_reference_count,
            b_l0_reference_count,
            profile_idc,
            level_idc,
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config.color.description();
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("H.265 encoder created successfully");
        Ok(encoder)
    }

    /// Profile IDC (used when rebuilding parameter sets).
    pub(super) fn profile_idc(&self) -> u32 {
        self.profile_idc
    }
}
