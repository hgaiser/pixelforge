//! Error types for PixelForge.

use thiserror::Error;

/// Main error type for PixelForge operations.
#[derive(Error, Debug)]
pub enum PixelForgeError {
    /// Vulkan instance creation failed.
    #[error("Failed to create Vulkan instance: {0}")]
    InstanceCreation(String),

    /// The device cannot do what was asked: it lacks a queue, an extension,
    /// a feature, a format or a profile. The same request can work on another
    /// device, or with another configuration on this one; nothing is wrong
    /// with the device itself.
    #[error("No suitable Vulkan physical device found: {0}")]
    NoSuitableDevice(String),

    /// The device was lost: a driver reset, a GPU hang, or the GPU going
    /// away. Nothing made from this context works any more. Drop the
    /// encoders, decoders and the context, and make a new context.
    #[error("The Vulkan device was lost")]
    DeviceLost,

    /// Vulkan logical device creation failed.
    #[error("Failed to create Vulkan device: {0}")]
    DeviceCreation(String),

    /// Video session creation failed.
    #[error("Failed to create video session: {0}")]
    VideoSessionCreation(String),

    /// Video session parameters creation failed.
    #[error("Failed to create video session parameters: {0}")]
    SessionParametersCreation(String),

    /// Memory allocation failed.
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Resource creation failed (images, buffers, fences, command pools, etc.).
    #[error("Failed to create resource: {0}")]
    ResourceCreation(String),

    /// Command buffer operation failed.
    #[error("Command buffer error: {0}")]
    CommandBuffer(String),

    /// Invalid input (dimensions, data size, format, etc.).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Codec not supported.
    #[error("Codec not supported: {0}")]
    CodecNotSupported(String),

    /// Synchronization primitive error.
    #[error("Synchronization error: {0}")]
    Synchronization(String),

    /// Query pool error.
    #[error("Query pool error: {0}")]
    QueryPool(String),

    /// The encoded bitstream overflowed the destination buffer.
    #[error("Encoder bitstream buffer overflow (wrote {written} bytes into {capacity} bytes)")]
    BufferOverflow {
        /// Number of bytes the encoder reported writing.
        written: usize,
        /// Capacity of the bitstream buffer in bytes.
        capacity: usize,
    },

    /// Generic Vulkan error.
    #[error("Vulkan error: {0}")]
    Vulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for PixelForgeError {
    fn from(result: ash::vk::Result) -> Self {
        Self::vk(result, |_| PixelForgeError::Vulkan(result))
    }
}

impl PixelForgeError {
    /// `wrap` the description of a failed Vulkan call, except a lost device,
    /// which is always [`Self::DeviceLost`].
    pub(crate) fn vk(result: ash::vk::Result, wrap: impl FnOnce(String) -> Self) -> Self {
        if result == ash::vk::Result::ERROR_DEVICE_LOST {
            Self::DeviceLost
        } else {
            wrap(result.to_string())
        }
    }
}

/// Result type for PixelForge operations.
pub type Result<T> = std::result::Result<T, PixelForgeError>;

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    #[test]
    fn a_lost_device_is_always_device_lost() {
        assert!(matches!(
            PixelForgeError::vk(
                vk::Result::ERROR_DEVICE_LOST,
                PixelForgeError::CommandBuffer
            ),
            PixelForgeError::DeviceLost
        ));
        assert!(matches!(
            PixelForgeError::from(vk::Result::ERROR_DEVICE_LOST),
            PixelForgeError::DeviceLost
        ));
        assert!(matches!(
            PixelForgeError::vk(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                PixelForgeError::CommandBuffer
            ),
            PixelForgeError::CommandBuffer(_)
        ));
    }
}
