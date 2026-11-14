variable "region" {
  description = "Where the bucket and the compile VMs live; the spot G-instance quota is per region."
  type        = string
  default     = "us-east-2"
}
