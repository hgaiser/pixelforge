use super::{Av1, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};

use crate::converter::InputFormat;
use crate::encoder::EncodeConfig;
use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, build_encoder_common, query_video_caps,
};
use crate::encoder::profile::EncodeProfile;
use crate::encoder::rgb::RgbConversionCaps;
use crate::error::Result;
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

impl Av1 {
    /// Create a new AV1 encoder.
    pub(crate) fn create(
        context: VideoContext,
        config: EncodeConfig,
        rgb_input: Option<InputFormat>,
    ) -> Result<CodecEncoder<Self>> {
        info!(
            "Creating AV1 encoder: {}x{}, pixel_format={:?}",
            config.dimensions.width, config.dimensions.height, config.pixel_format
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
            // slot on top of the active count, so cap active refs at 7 to keep
            // every DPB slot index in 0..=7 (the range the [u8; 8] order-hint
            // array and `1 << slot` refresh mask can represent).
            max_active_refs_cap: 7,
            bitstream_buffer_size,
            // Allow the shared infrastructure to fall back to a layered DPB
            // when the driver does not support separate reference images
            // (required for AMD RADV).
            allow_layered_dpb: true,
            rgb_caps,
        })?;
        let active_reference_count = init.active_reference_count;
        let common = init.common;

        let codec = Av1 {
            frame_num: 0,
            order_hint: 0,
            header_data: None,
            references: Vec::new(),
            active_reference_count,
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config.color.description();
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("AV1 encoder created successfully");
        Ok(encoder)
    }
}
