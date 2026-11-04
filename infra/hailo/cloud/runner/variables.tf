# run_id and suite_key end up in single-quoted shell assignments in user data,
# hence the restricted characters.

variable "run_id" {
  description = "The run's prefix under runs/ in the bucket."
  type        = string

  validation {
    condition     = can(regex("^[0-9A-Za-z-]+$", var.run_id))
    error_message = "run_id may contain only letters, digits and dashes."
  }
}

variable "suite_key" {
  description = "The Hailo DFC docker suite tarball's key in the bucket, under suite/."
  type        = string

  validation {
    condition     = can(regex("^suite/[0-9A-Za-z._-]+$", var.suite_key))
    error_message = "suite_key must be suite/<file name> of letters, digits, dots, dashes and underscores."
  }
}

variable "max_minutes" {
  description = "Minutes after boot at which the VM powers off and terminates, finished or not."
  type        = number
  default     = 90
}

# 8 vCPUs, 32 GB of RAM and a T4 (Turing, one of the GPU generations the Dataflow Compiler supports).
variable "instance_type" {
  type    = string
  default = "g4dn.2xlarge"
}
