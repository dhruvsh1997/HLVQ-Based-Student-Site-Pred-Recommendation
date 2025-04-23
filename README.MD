# 📊 Student Data Analysis using HLVQ and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An advanced analysis of student data using Hierarchical Learning Vector Quantization (HLVQ) and other machine learning techniques to predict site selection changes and provide personalized academic recommendations.

![Student Analysis Banner](https://via.placeholder.com/800x200?text=Student+Data+Analysis+Project)

## 📋 Table of Contents

- [Introduction](#-introduction)
- [What is HLVQ?](#-what-is-hlvq)
- [Project Components](#-project-components)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Machine Learning Models](#machine-learning-models)
    - [Neural Network](#neural-network)
    - [HLVQ](#hlvq)
    - [AdaBoost](#adaboost)
  - [Site Suggestion Logic](#site-suggestion-logic)
  - [Grade Assignment](#grade-assignment)
  - [Visualizations](#visualizations)
- [Results and Evaluation](#-results-and-evaluation)
- [Conclusion](#-conclusion)
- [Requirements](#-requirements)
- [References](#-references)
- [Contact](#-contact)

## 🚀 Introduction

This project analyzes student academic data to predict whether students might benefit from changing their academic stream or program (site selection) and recommends alternative sites based on their performance. The core analysis is performed in the Jupyter Notebook `HLVQ Student Data Analysis Calc.ipynb`, which employs various machine learning techniques with a special focus on **Hierarchical Learning Vector Quantization (HLVQ)**.

The dataset (`hcslMREL.csv`) contains comprehensive student information including:
- Academic scores across different subjects
- Current site selections
- Whether they suggest a change

This analysis aims to provide educational institutions with actionable insights by predicting student behavior and personalizing recommendations to optimize academic outcomes.

## 🧠 What is HLVQ?

**Hierarchical Learning Vector Quantization (HLVQ)** is an advanced extension of Learning Vector Quantization (LVQ), a specialized type of artificial neural network designed for classification tasks. 

| LVQ | HLVQ |
|-----|------|
| Represents classes with prototypes (reference vectors) | Enhances LVQ with hierarchical structure |
| Adjusts prototypes during training | Organizes prototypes to capture complex relationships |
| Effective for simple classifications | Better suited for complex, multi-layered data patterns |

In this project, we implement a custom HLVQ algorithm that classifies students based on multiple features to predict whether they might suggest a site change. The algorithm utilizes parameters like learning rate, decay rate, and maximum epochs to adjust prototypes hierarchically, making it a uniquely powerful component of our analysis.

## 🔍 Project Components

### Data Loading and Preprocessing

- **Dataset Source**: `hcslMREL.csv`
- **Key Variables**:
  - Academic scores (science, math, social studies)
  - Site selections
  - `Sugg_Change` (target variable)
  
- **Preprocessing Steps**:
  - Converting `Sugg_Change` to binary (1 for 'Yes', 0 for 'No')
  - Feature selection
  - Train-test split (80% training, 20% testing)

### Machine Learning Models

#### Neural Network

```
INPUT LAYER → HIDDEN LAYER (10 neurons) → HIDDEN LAYER (15 neurons) → OUTPUT LAYER (sigmoid)
```

- **Architecture**: Feedforward neural network with TensorFlow/Keras
- **Training Configuration**:
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
  - Epochs: Adjustable based on convergence
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

#### HLVQ

- **Custom Implementation Parameters**:
  - Learning rate (a): 0.3
  - Decay rate (b): 0.2
  - Maximum epochs (max_ep): 100
  - Minimum learning rate (min_a): 0.001
  - Hierarchical adjustment factor (e): 0.3

- **Process Flow**:
  1. Initialize prototype vectors
  2. For each epoch:
     - Present training samples
     - Find closest prototype
     - Update prototypes based on correctness
     - Apply hierarchical adjustments
     - Decay learning rate
  3. Classification based on final prototypes

#### AdaBoost

- **Configuration**: 
  - 100 estimators (decision trees)
  - Standard scikit-learn implementation
- **Purpose**: Provides baseline comparison for HLVQ and neural network performance

### Site Suggestion Logic

For students with `Sugg_Change == 'Yes'`, we recommend alternative sites based on subject performance:

| Condition | Recommendation |
|-----------|----------------|
| Science mark ≥ 70, others < 70 | Sci&Tech |
| Social Studies mark ≥ 70, others < 70 | Arts |
| Math mark ≥ 70, others < 70 | Comm&Mgmt |

This logic personalizes site recommendations to align with students' academic strengths.

### Grade Assignment

Grades are assigned based on the following criteria:

| Score Range | Grade |
|-------------|-------|
| 85-100 | A |
| 75-85 | B |
| 65-75 | C |
| 55-65 | D |
| 40-55 | E |

A new `Grade` column is added to the dataset for clearer performance categorization.

### Visualizations

The notebook generates several visualizations saved as PNG files:

| Visualization Type | Content | Filename |
|-------------------|---------|----------|
| Bar Graphs | Student distribution across sites | mpl_plot_1.png |
| | Suggestion ratios | mpl_plot_2.png |
| Pie Charts | Percentage of students suggesting change | mpl_plot_7.png |
| | Site selection shares | mpl_plot_3.png, mpl_plot_5.png |
| Histograms | Distribution of percentage effects | mpl_plot_6.png |
| Box Plots | Student numbers across new sites | mpl_plot_9.png |
| Density Plot | Score distribution with normal overlay | - |

## 📈 Results and Evaluation

Our analysis provides comprehensive insights into student site preferences and potential changes:

- **Neural Network Performance**: ~89.8% accuracy
- **HLVQ Metrics**: 
  - Custom performance calculation
  - Slight adjustments for optimization (e.g., small constant subtraction from recall)
- **AdaBoost Benchmark**: Comparable performance metrics
- **Site Redistribution**: Visualizations show how students would be redistributed to new sites based on academic strengths
- **Grade Distribution**: Density plots illustrate the spread of student scores across grade categories

## 🎓 Conclusion

This project successfully demonstrates the application of HLVQ alongside traditional machine learning techniques to analyze student data, predict site change suggestions, and provide personalized academic recommendations. The custom HLVQ implementation offers a specialized approach to classification that captures complex patterns in educational data.

The visualizations provide clear insights into student performance trends and preferences, making the results accessible to educational administrators and policymakers.

**Future Improvements**:
- Refining the HLVQ algorithm parameters
- Exploring additional student features
- Validating site suggestions with educational experts
- Implementing longitudinal analysis to track changes over time

## ⚙️ Requirements

To run the notebook, install the following Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow scipy
```

## 📚 References

- [Learning Vector Quantization](https://en.wikipedia.org/wiki/Learning_vector_quantization)
- [Hierarchical Learning Vector Quantization](https://ieeexplore.ieee.org/document/548)
- Additional references available upon request

## 📬 Contact

For questions, contributions, or collaboration opportunities, please contact:

[Your Name/Email]

---

<p align="center">
  <i>Empowering educational decisions through data science and machine learning</i>
</p>