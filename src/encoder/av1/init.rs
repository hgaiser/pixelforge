use super::{Av1, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};
use crate::encoder::references::ReferenceSet;

use crate::converter::RgbFormat;
use crate::encoder::EncodeConfig;
use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, build_encoder_common, check_quality_level, query_video_caps,
};
use crate::encoder::profile::EncodeProfile;
use crate::encoder::rgb::RgbConversionCaps;
use crate::error::Result;
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::{debug, info};

impl Av1 {
    /// Create a new AV1 encoder.
    pub(crate) fn create(
        context: VideoContext,
        config: EncodeConfig,
        rgb_input: Option<RgbFormat>,
    ) -> Result<CodecEncoder<Self>> {
        info!(
            "Creating AV1 encoder: {}x{}, chroma_format={:?}",
            config.dimensions.width, config.dimensions.height, config.chroma_format
        );

        let profile = EncodeProfile::new(&config, rgb_input.is_some())?;
        let profile_info = profile.info();

        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        // Query device capabilities with the AV1-specific capability struct
        // chained in (required by the driver).
        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut av1_caps);
        let mut rgb_caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        if rgb_input.is_some() {
            capabilities = capabilities.push(&mut rgb_caps);
        }
        let caps = query_video_caps(&context, profile_info, &mut capabilities)?;
        check_quality_level(&config, encode_caps.max_quality_levels)?;
        let rgb_caps = rgb_input.map(|_| RgbConversionCaps::from(&rgb_caps));

        let init = build_encoder_common(CommonInitRequest {
            context: &context,
            config: &config,
            profile,
            rgb_input,
            caps: &caps,
            align_unit: SUPERBLOCK_SIZE,
            // AV1 has NUM_REF_FRAMES = 8 DPB slots, and `refresh_frame_flags`
            // is an 8-bit syntax element. `build_encoder_common` adds one setup
            // slot on top of the window, so cap the window at 7 to keep every
            // DPB slot index in 0..=7 (the range the [u8; 8] order-hint array
            // and `1 << slot` refresh mask can represent).
            max_reference_window: 7,
            // Inter frames use single-reference prediction, so each may name
            // at most this many references, and only names in the mask.
            max_frame_references: (av1_caps.max_single_reference_count as usize)
                .min(av1_caps.single_reference_name_mask.count_ones() as usize),
            bitstream_buffer_size,
            // Allow the shared infrastructure to fall back to a layered DPB
            // when the driver does not support separate reference images
            // (required for AMD RADV).
            allow_layered_dpb: true,
            rgb_caps,
            b_frame_support: crate::encoder::bframes::av1_support(
                context.device_properties().vendor_id,
                &av1_caps,
                caps.max_active_reference_pictures,
            ),
        })?;
        let reference_window = init.reference_window;
        let active_reference_count = init.active_reference_count;
        // The past references of a frame that predicts from both directions:
        // what the device allows for the names before `BWDREF`, less the one
        // future reference.
        let bidirectional_past_count = av1_caps
            .max_bidirectional_compound_group1_reference_count
            .min(
                av1_caps
                    .max_bidirectional_compound_reference_count
                    .saturating_sub(1),
            )
            .min(init.session_active_references.saturating_sub(1));
        debug!(
            "AV1 bidirectional prediction: {} references ({} past names, {} future names), \
             name mask {:#09b}",
            av1_caps.max_bidirectional_compound_reference_count,
            av1_caps.max_bidirectional_compound_group1_reference_count,
            av1_caps.max_bidirectional_compound_group2_reference_count,
            av1_caps.bidirectional_compound_reference_name_mask
        );
        let common = init.common;

        let codec = Av1 {
            frame_num: 0,
            header_data: None,
            references: ReferenceSet::new(reference_window),
            active_reference_count,
            single_reference_name_mask: av1_caps.single_reference_name_mask,
            bidirectional_reference_name_mask: av1_caps.bidirectional_compound_reference_name_mask,
            bidirectional_past_count,
            coded_size: super::coded_size(
                config.dimensions.width,
                config.dimensions.height,
                av1_caps.coded_picture_alignment,
            ),
        };
        debug!(
            "AV1 coded picture alignment {}x{}: frames are coded at {}x{}",
            av1_caps.coded_picture_alignment.width,
            av1_caps.coded_picture_alignment.height,
            codec.coded_size.width,
            codec.coded_size.height
        );

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config.color.description();
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("AV1 encoder created successfully");
        Ok(encoder)
    }
}
