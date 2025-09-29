# STRIDE Threat Model: MNIST CNN Classifier Application

## Overview
This document applies the STRIDE framework to analyze threats for the MNIST CNN classifier application, which trains and evaluates a convolutional neural network on the MNIST dataset using TensorFlow.

## STRIDE Categories

### 1. Spoofing
- **Threats:**
  - Impersonation of legitimate users if authentication is added in future.
  - Use of tampered or fake MNIST dataset files.
- **Mitigations:**
  - Validate dataset integrity (checksums, hashes).
  - Implement authentication for multi-user or networked deployments.

### 2. Tampering
- **Threats:**
  - Modification of training or test data (e.g., adversarial samples, data poisoning).
  - Corruption of model files or source code.
- **Mitigations:**
  - Restrict file permissions and access.
  - Use version control for code and data.
  - Validate input data before use.

### 3. Repudiation
- **Threats:**
  - Users deny actions such as training, evaluation, or data modification.
  - Lack of audit trails for model changes or data access.
- **Mitigations:**
  - Implement logging of key actions and events.
  - Maintain audit trails for data and model changes.

### 4. Information Disclosure
- **Threats:**
  - Leakage of sensitive data (if private datasets are used).
  - Exposure of model internals or training data via logs or saved files.
- **Mitigations:**
  - Limit access to logs and output files.
  - Sanitize logs to avoid leaking sensitive information.
  - Store models and data in secure locations.

### 5. Denial of Service
- **Threats:**
  - Resource exhaustion (CPU, memory) due to large or malformed datasets.
  - Infinite loops or hangs from malicious data or code.
- **Mitigations:**
  - Monitor and limit resource usage.
  - Validate and sanitize all input data.
  - Use exception handling to prevent crashes.

### 6. Elevation of Privilege
- **Threats:**
  - Unauthorized users gain access to model files, code, or data.
  - Exploitation of vulnerabilities in dependencies to gain elevated privileges.
- **Mitigations:**
  - Enforce strict access controls on files and directories.
  - Regularly update and patch dependencies.
  - Run application with least privilege required.

## Summary Table
| STRIDE Category         | Example Threats                                  | Mitigations                       |
|------------------------|--------------------------------------------------|-----------------------------------|
| Spoofing               | Fake dataset, user impersonation                 | Dataset validation, authentication|
| Tampering              | Data/model corruption, adversarial samples       | Access control, validation        |
| Repudiation            | Denial of actions, lack of audit                 | Logging, audit trails             |
| Information Disclosure | Data/model leakage, log exposure                 | Secure storage, log sanitization  |
| Denial of Service      | Resource exhaustion, infinite loops              | Resource limits, input validation |
| Elevation of Privilege | Unauthorized access, dependency exploits         | Least privilege, patching         |

---
This STRIDE threat model should be reviewed and updated as the application evolves or its deployment context changes.
