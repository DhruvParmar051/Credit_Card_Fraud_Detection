# **Credit Card Fraud Detection 🚀**

## **📌 Overview**

Credit card fraud is a major concern in the financial industry. This project applies **Machine Learning (ML)** techniques to detect fraudulent transactions based on various features extracted from transaction data.

## **📁 Project Structure**

```plaintext
Credit-Card-Fraud-Detection/
│── notebook/
│   ├── EDA STUDENT PERFORMANCE.ipynb      # Exploratory Data Analysis (EDA)
│   ├── MODEL TRAINING.ipynb               # Model Training Notebook
│
│── src/
│   ├── components/                        # Core ML model components
│   ├── pipeline/                          # Data processing pipeline
│   ├── __init__.py                        # Package initialization
│   ├── exception.py                       # Custom exception handling
│   ├── logger.py                          # Logging module
│   ├── utils.py                           # Utility functions
│
│── templates/
│   ├── home.html                          # Home Page HTML
│   ├── index.html                         # Main Web Interface
│
│── .gitignore
│── README.md                              # Project Documentation
│── application.py                         # Flask API for model deployment
│── requirements.txt                        # Dependencies List
│── setup.py         

```

---

## **⚡ Features**

- **Exploratory Data Analysis (EDA)** using **Pandas, Matplotlib, and Seaborn**.
- **Feature Engineering** to preprocess the dataset.
- **Machine Learning Model Training** using **Scikit-Learn**.
- **Hyperparameter Tuning** for model optimization.
- **Web App Deployment** using **Flask**.
- **Logging and Exception Handling** for better debugging.

---

## **📊 Dataset**

The dataset is sourced from **Kaggle's Credit Card Fraud Detection** dataset.

🔗 **Dataset Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### **💾 Download Dataset Automatically**

Follow these steps to download the dataset from Kaggle:

### **Step 1: Install the Kaggle API**

```sh
pip install kaggle
```

### **Step 2: Configure Kaggle API**

- Go to [Kaggle API Settings](https://www.kaggle.com/account)
- Scroll down to **API**, click on **Create New API Token**.
- It will download a file named `kaggle.json`.
- Move this file to the appropriate directory:

  ```sh
  
  mkdir -p ~/.kaggle
  mv kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```

### **Step 3: Download the Dataset**

Run the following command in your project directory:

```sh
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

This will download and extract the dataset into the `data/` folder.

---

## **🔧 Installation**

### **Step 1: Clone the Repository**

```sh
git clone https://github.com/DHRUV051/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### **Step 2: Create a Virtual Environment**

```sh
python -m venv env
```

Activate the environment:

- **Windows**: `env\Scriptsctivate`
- **Mac/Linux**: `source env/bin/activate`

### **Step 3: Install Dependencies**

```sh
pip install -r requirements.txt
```

---

## **🚀 Usage**

### **1️⃣ Train the Model**

Run the **`MODEL TRAINING.ipynb`** Jupyter Notebook to train your model.

### **2️⃣ Run the Flask Web App**

After training the model, deploy it using Flask:

```sh
python application.py
```

Then, open **<http://127.0.0.1:5000/>** in your browser.

---

## **📌 Future Improvements**

✅ **Deep Learning Integration** (e.g., LSTMs, Autoencoders)  
✅ **Real-time Fraud Detection System**  
✅ **Integration with Cloud Platforms (AWS, GCP, Azure)**  

---

## **💡 Contributing**

If you want to improve the project, feel free to submit a **pull request**. Make sure to discuss any major changes in an issue first.

---

## **📜 License**

This project is open-source under the **MIT License**.

---

## **📩 Contact**

🔗 **Author**: Dhruv Parmar  
📧 **Email**: <dhruvparmar051@gmail.com>  
🔗 **GitHub**: [DHRUV051](https://github.com/DHRUV051)

---
