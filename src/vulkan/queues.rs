//! Which queues pixelforge submits to on a device it did not create.
//!
//! A `VkQueue` must be externally synchronized: two threads may not submit to
//! it at once. A context that creates its own device owns every queue on it,
//! so the question never comes up. A context adopted from a caller's device
//! shares that device with the caller, and by default takes queue 0 of each
//! family it needs, which is very often the queue the caller renders and
//! presents on. Submitting there from another thread is a data race.
//!
//! [`DeviceQueue`] lets the caller say exactly which queue each role uses, so it
//! can create spare queues for pixelforge and keep its own to itself.

use crate::error::{PixelForgeError, Result};
use ash::vk;

/// One queue on a device: a queue family and an index within it, as passed to
/// `vkGetDeviceQueue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceQueue {
    /// Queue family index.
    pub family: u32,
    /// Index of the queue within its family. The caller must have created at
    /// least `index + 1` queues in `family`.
    pub index: u32,
}

impl DeviceQueue {
    /// Queue `index` of `family`.
    pub fn new(family: u32, index: u32) -> Self {
        Self { family, index }
    }
}

/// The queue family pixelforge would pick for each role it may use.
///
/// Returned as part of [`DeviceRequirements`](super::DeviceRequirements). A
/// role that the requested work does not need is `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRoles {
    /// Video encode submissions.
    pub encode: Option<u32>,
    /// Video decode submissions.
    pub decode: Option<u32>,
    /// Uploads and readback copies.
    pub transfer: u32,
    /// The colour converter's compute dispatches.
    pub compute: u32,
}

impl QueueRoles {
    /// The distinct families, in a stable order.
    pub fn unique_families(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for f in [
            self.encode,
            self.decode,
            Some(self.transfer),
            Some(self.compute),
        ]
        .into_iter()
        .flatten()
        {
            if !out.contains(&f) {
                out.push(f);
            }
        }
        out
    }
}

/// Queues the caller chose for an adopted context, one per role. `None` means
/// queue 0 of the family pixelforge selects.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct QueueOverrides {
    pub encode: Option<DeviceQueue>,
    pub decode: Option<DeviceQueue>,
    pub transfer: Option<DeviceQueue>,
    pub compute: Option<DeviceQueue>,
}

/// Resolve the queue for one role: the caller's choice if there is one,
/// checked against what the role needs, otherwise queue 0 of `default_family`.
///
/// The index cannot be checked against how many queues the caller actually
/// created, since Vulkan does not report that; it is checked against how many
/// the family offers, which catches the mistakes that can be caught.
pub(crate) fn resolve(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    role: &str,
    chosen: Option<DeviceQueue>,
    default_family: u32,
    needs: vk::QueueFlags,
) -> Result<DeviceQueue> {
    let Some(queue) = chosen else {
        return Ok(DeviceQueue::new(default_family, 0));
    };
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let Some(props) = families.get(queue.family as usize) else {
        return Err(PixelForgeError::InvalidInput(format!(
            "{role} queue: family {} does not exist (the device has {})",
            queue.family,
            families.len()
        )));
    };
    if !family_can(props.queue_flags, needs) {
        return Err(PixelForgeError::InvalidInput(format!(
            "{role} queue: family {} has {:?}, which does not include {:?}",
            queue.family, props.queue_flags, needs
        )));
    }
    if queue.index >= props.queue_count {
        return Err(PixelForgeError::InvalidInput(format!(
            "{role} queue: index {} is out of range, family {} has {} queue(s)",
            queue.index, queue.family, props.queue_count
        )));
    }
    Ok(queue)
}

/// Whether a family with `flags` can do `needs`. Graphics and compute families
/// support transfer operations whether or not they advertise it.
fn family_can(flags: vk::QueueFlags, needs: vk::QueueFlags) -> bool {
    let mut effective = flags;
    if flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
        effective |= vk::QueueFlags::TRANSFER;
    }
    effective.contains(needs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphics_and_compute_imply_transfer() {
        assert!(family_can(
            vk::QueueFlags::GRAPHICS,
            vk::QueueFlags::TRANSFER
        ));
        assert!(family_can(
            vk::QueueFlags::COMPUTE,
            vk::QueueFlags::TRANSFER
        ));
        assert!(!family_can(
            vk::QueueFlags::VIDEO_ENCODE_KHR,
            vk::QueueFlags::TRANSFER
        ));
        assert!(!family_can(
            vk::QueueFlags::TRANSFER,
            vk::QueueFlags::COMPUTE
        ));
    }

    #[test]
    fn unique_families_keeps_first_occurrence_order() {
        let roles = QueueRoles {
            encode: Some(4),
            decode: None,
            transfer: 1,
            compute: 4,
        };
        assert_eq!(roles.unique_families(), vec![4, 1]);
    }
}
