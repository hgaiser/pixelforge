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

    /// Invalid input (dimensions, data size, format, etc.).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The encoded bitstream overflowed the destination buffer.
    #[error("Encoder bitstream buffer overflow (wrote {written} bytes into {capacity} bytes)")]
    BufferOverflow {
        /// Number of bytes the encoder reported writing.
        written: usize,
        /// Capacity of the bitstream buffer in bytes.
        capacity: usize,
    },

    /// A Vulkan call failed. A lost device is [`Self::DeviceLost`] instead.
    #[error("{call} failed: {result}")]
    Vulkan {
        /// The Vulkan function that failed.
        call: &'static str,
        /// What it returned.
        result: ash::vk::Result,
    },

    /// The driver reported success, but what it returned cannot be used,
    /// such as an encode that wrote no bytes.
    #[error("Driver error: {0}")]
    Driver(String),

    /// An operating system call failed.
    #[error("{what} failed: {source}")]
    Io {
        /// What failed.
        what: &'static str,
        /// The error.
        #[source]
        source: std::io::Error,
    },

    /// The encoder was dropped before the frame was read back.
    #[error("Encode cancelled: the encoder shut down before the frame was read back")]
    Cancelled,
}

impl PixelForgeError {
    /// The error for `result` from the Vulkan function `call`: a lost device
    /// is always [`Self::DeviceLost`].
    pub(crate) fn vulkan(call: &'static str, result: ash::vk::Result) -> Self {
        if result == ash::vk::Result::ERROR_DEVICE_LOST {
            Self::DeviceLost
        } else {
            Self::Vulkan { call, result }
        }
    }

    /// [`Self::vulkan`] for `map_err`.
    pub(crate) fn vk(call: &'static str) -> impl FnOnce(ash::vk::Result) -> Self {
        move |result| Self::vulkan(call, result)
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
            PixelForgeError::vulkan("vkQueueSubmit", vk::Result::ERROR_DEVICE_LOST),
            PixelForgeError::DeviceLost
        ));
        assert!(matches!(
            PixelForgeError::vk("vkQueueSubmit")(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            PixelForgeError::Vulkan {
                call: "vkQueueSubmit",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            }
        ));
    }
}
