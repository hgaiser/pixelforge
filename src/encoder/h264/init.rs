use super::{H264, MB_SIZE, ReferenceInfo};

use crate::converter::RgbFormat;
use crate::encoder::EncodeConfig;
use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, build_encoder_common, query_video_caps,
};
use crate::encoder::profile::EncodeProfile;
use crate::encoder::profile::h264_profile_idc;
use crate::encoder::resources::MIN_BITSTREAM_BUFFER_SIZE;
use crate::encoder::rgb::RgbConversionCaps;
use crate::error::Result;
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::{debug, info};

impl H264 {
    /// Create a new H.264 encoder.
    pub(crate) fn create(
        context: VideoContext,
        config: EncodeConfig,
        rgb_input: Option<RgbFormat>,
    ) -> Result<CodecEncoder<Self>> {
        info!(
            "Creating H.264 encoder: {}x{}, chroma_format={:?}",
            config.dimensions.width, config.dimensions.height, config.chroma_format
        );

        // Select the profile based on chroma subsampling.
        let profile_idc = h264_profile_idc(config.chroma_format);

        let profile = EncodeProfile::new(&config, rgb_input.is_some())?;
        let profile_info = profile.info();

        // Query device capabilities with the H.264-specific capability struct
        // chained in (required by the driver).
        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut h264_caps);
        let mut rgb_caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        if rgb_input.is_some() {
            capabilities = capabilities.push(&mut rgb_caps);
        }
        let caps = query_video_caps(&context, profile_info, &mut capabilities)?;
        let rgb_caps = rgb_input.map(|_| RgbConversionCaps::from(&rgb_caps));

        // The driver's preferred entropy mode is H.264-specific (some require
        // CAVLC for High 4:4:4 Predictive). Only after the capability query,
        // which fails for an unsupported profile: a quality level query for one
        // is invalid (VUID-VkPhysicalDeviceVideoEncodeQualityLevelInfoKHR-pVideoProfile-08259).
        let preferred_entropy_cabac = query_preferred_entropy_cabac(&context, profile_info);

        // Scale the bitstream buffer by resolution so a single frame can't overflow (matches AV1).
        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        let init = build_encoder_common(CommonInitRequest {
            context: &context,
            config: &config,
            profile,
            rgb_input,
            caps: &caps,
            align_unit: MB_SIZE,
            max_reference_window: 32,
            max_frame_references: h264_caps.max_p_picture_l0_reference_count as usize,
            bitstream_buffer_size,
            allow_layered_dpb: true,
            rgb_caps,
            b_frame_support: crate::encoder::bframes::h264_support(
                &h264_caps,
                caps.max_active_reference_pictures,
            ),
        })?;
        let reference_window = init.reference_window;
        let active_reference_count = init.active_reference_count;
        let common = init.common;

        let codec = H264 {
            frame_num_syntax: 0,
            idr_pic_id: 0,
            l0_references: Vec::new(),
            reference_window,
            active_reference_count,
            pending_unmark_frame_nums: Vec::new(),
            profile_idc,
            preferred_entropy_cabac,
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config.color.description();
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("H.264 encoder created successfully");
        Ok(encoder)
    }

    /// Profile IDC (used when rebuilding parameter sets).
    pub(super) fn profile_idc(&self) -> u32 {
        self.profile_idc
    }

    /// Preferred entropy coding mode (CABAC if true, CAVLC otherwise).
    pub(super) fn preferred_entropy_cabac(&self) -> bool {
        self.preferred_entropy_cabac
    }

    /// How many of the newest references one frame predicts from.
    pub(super) fn active_reference_count(&self) -> u32 {
        self.active_reference_count
    }

    /// How many references stay in the DPB.
    pub(super) fn reference_window(&self) -> u32 {
        self.reference_window
    }

    /// The references this frame predicts from: the newest held ones.
    pub(super) fn used_references(&self) -> &[ReferenceInfo] {
        let n = self
            .l0_references
            .len()
            .min(self.active_reference_count as usize);
        &self.l0_references[..n]
    }
}

/// Query the driver's preferred H.264 entropy coding mode for quality level 0.
/// Defaults to CABAC if the query is unsupported.
fn query_preferred_entropy_cabac(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
) -> bool {
    let video_encode_instance =
        ash::khr::video_encode_queue::Instance::load(context.entry(), context.instance());
    let mut h264_quality_level_properties = vk::VideoEncodeH264QualityLevelPropertiesKHR::default();
    let mut quality_level_properties = vk::VideoEncodeQualityLevelPropertiesKHR::default()
        .push(&mut h264_quality_level_properties);
    let quality_level_info = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default()
        .video_profile(profile_info)
        .quality_level(0);
    let result = unsafe {
        (video_encode_instance
            .fp()
            .get_physical_device_video_encode_quality_level_properties_khr)(
            context.physical_device(),
            &quality_level_info,
            &mut quality_level_properties,
        )
    };
    if result == vk::Result::SUCCESS {
        debug!(
            "H.264 quality level 0: preferredStdEntropyCodingModeFlag={}",
            h264_quality_level_properties.preferred_std_entropy_coding_mode_flag
        );
        h264_quality_level_properties.preferred_std_entropy_coding_mode_flag != 0
    } else {
        debug!("Quality level query failed ({result:?}); defaulting to CABAC");
        true
    }
}
