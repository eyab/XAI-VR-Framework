# Verification and Explanation Framework for Cyber Defense Agents

## Overview
This project focuses on designing and implementing a **Verification and Explanation Framework** for Blue agent policies in the CybORG cybersecurity environment. The framework aims to enhance the **transparency**, **correctness**, and **resilience** of autonomous decision-making systems tasked with defending networks against adversarial attacks.

---

## Key Objectives
- **Explainability**: Use SHAP (SHapley Additive exPlanations) to provide local and global explanations for the agent’s actions, offering insights into feature contributions and decisions.
- **Verification**: Ensure the correctness and reliability of Blue agent policies using formal verification techniques such as **model checking**.
- **Resilience**: Validate that the Blue agent’s policies are robust against adversarial manipulation and unintended consequences.

---

## Features

### **Explainability Framework**
- Generates detailed SHAP-based explanations tailored for system designers and end users.
- Provides actionable insights into feature contributions for the Blue agent's actions.

### **Verification Framework**
- Uses formal methods (e.g., LTL/PCTL) to validate policy correctness.
- Evaluates the agent’s behavior across all possible states to ensure compliance with desired outcomes.

### **Resilience Testing**
- Simulates adversarial scenarios to assess policy robustness.
- Measures the agent's adaptability to challenging conditions and unexpected behaviors.

---

## How It Works

### **Data Collection**
- The CybORG environment provides observations, actions, and rewards during simulations.

### **Explainability**
- SHAP values are computed to quantify the influence of environmental features on the Blue agent’s decisions.
- Contextual explanations are generated to align actions with observed threats.

### **Verification**
- Formal verification techniques ensure the agent’s policies are aligned with predefined safety and performance requirements.
- LTL/PCTL formulas are used to verify critical properties such as "no backdoors" and "correct behavior under all scenarios."

### **Integration**
- Explainability and verification outputs are combined to create a cohesive framework for evaluating Blue agent policies.

---
