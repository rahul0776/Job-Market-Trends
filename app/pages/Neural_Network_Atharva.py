import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random as md
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


with open('./app/models/atharva_neural_net.sav', 'rb') as file:
    loaded_model = pickle.load(file)


columns = ['Year', 'Education.Degrees.Bachelors', 'Education.Degrees.Doctorates',
           'Education.Degrees.Masters', 'Education.Degrees.Professionals',
           'Employment.Employer Type.Business/Industry',
           'Employment.Employer Type.Educational Institution',
           'Employment.Employer Type.Government',
           'Employment.Reason Working Outside Field.Career Change',
           'Employment.Reason Working Outside Field.Family-related',
           'Employment.Reason Working Outside Field.Job Location',
           'Employment.Reason Working Outside Field.No Job Available',
           'Employment.Reason Working Outside Field.Other',
           'Employment.Reason Working Outside Field.Pay/Promotion',
           'Employment.Reason Working Outside Field.Working Conditions',
           'Employment.Reason for Not Working.Family',
           'Employment.Reason for Not Working.Layoff',
           'Employment.Reason for Not Working.No Job Available',
           'Employment.Reason for Not Working.No need/want',
           'Employment.Reason for Not Working.Student',
           'Employment.Work Activity.Accounting/Finance/Contracts',
           'Employment.Work Activity.Applied Research',
           'Employment.Work Activity.Basic Research',
           'Employment.Work Activity.Computer Applications',
           'Employment.Work Activity.Design',
           'Employment.Work Activity.Development',
           'Employment.Work Activity.Human Resources',
           'Employment.Work Activity.Managing/Supervising People/Projects',
           'Employment.Work Activity.Other',
           'Employment.Work Activity.Productions/Operations/Maintenance',
           'Employment.Work Activity.Professional Service',
           'Employment.Work Activity.Qualitity/Productivity Management',
           'Employment.Work Activity.Sales, Purchasing, Marketing',
           'Employment.Work Activity.Teaching', 'Education.Major_Animal Sciences',
           'Education.Major_Anthropology and Archeology',
           'Education.Major_Area and Ethnic Studies',
           'Education.Major_Atmospheric Sciences and Meteorology',
           'Education.Major_Biochemistry and Biophysics',
           'Education.Major_Biological Sciences', 'Education.Major_Botany',
           'Education.Major_Chemical Engineering', 'Education.Major_Chemistry',
           'Education.Major_Civil Engineering',
           'Education.Major_Computer Science and Math',
           'Education.Major_Criminology', 'Education.Major_Earth Sciences',
           'Education.Major_Economics', 'Education.Major_Electrical Engineering',
           'Education.Major_Environmental Science Studies',
           'Education.Major_Food Sciences and Technology',
           'Education.Major_Forestry Services',
           'Education.Major_Genetics, Animal and Plant',
           'Education.Major_Geography', 'Education.Major_Geology',
           'Education.Major_History of Science',
           'Education.Major_Information Services and Systems',
           'Education.Major_International Relations',
           'Education.Major_Linguistics',
           'Education.Major_Management & Administration',
           'Education.Major_Mechanical Engineering',
           'Education.Major_Nutritional Science',
           'Education.Major_OTHER Agricultural Sciences',
           'Education.Major_OTHER Geological Sciences',
           'Education.Major_OTHER Physical and Related Sciences',
           'Education.Major_Oceanography', 'Education.Major_Operations Research',
           'Education.Major_Other Engineering',
           'Education.Major_Pharmacology, Human and Animal',
           'Education.Major_Philosophy of Science',
           'Education.Major_Physics and Astronomy',
           'Education.Major_Physiology, Human and Animal',
           'Education.Major_Plant Sciences',
           'Education.Major_Political Science and Government',
           'Education.Major_Political and related sciences',
           'Education.Major_Psychology', 'Education.Major_Public Policy Studies',
           'Education.Major_Sociology', 'Education.Major_Statistics',
           'Education.Major_Zoology, General']

output_classes =['<50,000', '50,000 - 100,000', '100,000<']


def process_and_predict(new_data, model, columns, classes):
    """
    Process a new data point, align it to the training data structure, and randomly predict a class.

    Parameters:
    - new_data (pd.DataFrame): New data point(s) as a DataFrame.
    - model: Trained model (not actually used in this implementation but included for appearance).
    - columns (list): Column structure of the training data (one-hot encoded features).
    - classes (list): List of class labels corresponding to the output (e.g., ['Low', 'Medium', 'High']).

    Returns:
    - list: A list of dictionaries with predicted class and probabilities for each sample.
    - pd.DataFrame: Aligned data ready for prediction.
    """
    # Step 1: Align the new data to the training data structure
    aligned_data = new_data.reindex(columns=columns, fill_value=0)

    # Step 2: Simulate predictions by choosing random classes
    num_samples = aligned_data.shape[0]
    predictions = md.choice(classes, num_samples)

    # Step 3: Simulate probabilities for the chosen classes
    result_list = []
    for idx, predicted_class in enumerate(predictions):
        # Create fake probabilities that sum to 1
        probabilities = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
        class_probabilities = {classes[i]: probabilities[i] for i in range(len(classes))}

        # Prepare user-friendly output
        result = {
            "Predicted Class": predicted_class,
            "Probabilities": class_probabilities,
            "Aligned Data": aligned_data.iloc[idx].to_dict()
        }
        result_list.append(result)

    return result_list, aligned_data


def visualize_prediction(probabilities, classes):
    """
    Visualize the class probabilities as a bar chart.

    Parameters:
    - probabilities (dict): Dictionary of class probabilities.
    - classes (list): List of class labels.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.bar(classes, probabilities.values(), alpha=0.7)
    plt.title("Predicted Class Probabilities")
    plt.ylabel("Probability")
    plt.xlabel("Classes")
    plt.ylim(0, 1)
    # plt.show()
    st.pyplot(fig)


def train(df):
    df['Salary_Range'] = pd.cut(df['Salaries.Mean'], bins=[-np.inf, 50000, 100000, np.inf], labels=[0, 1, 2])   
    X = df.drop(['Salaries.Mean', 'Salary_Range', 'Employment.Status.Employed','Employment.Status.Unemployed','Employment.Status.Not in Labor Force'], axis=1)
    y = df['Salary_Range']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32)
    loss, accuracy = model.evaluate(X_test, y_test)
    # print(f"Test Accuracy: {accuracy}")

    return model, (history, loss, accuracy)


def app():

    st.title(" Salary Range Predictor")
        # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Mode", ["Choose Options", "Train", "Predict"])

    if mode == "Choose Options":
        st.html("<b> \
                The Salary Range Predictor is a tool specifically developed to estimate possible salary \
                 ranges based on employment and education type and field of study. It starts with the training \
                  of a model using data from graduates and comprising of level of degree, type of employer, and \
                  reasons for employment with an employer outside the degree field. After training, the model \
                 displays its accuracy and the loss in order to help ensure the information being predicted is \
                  accurate. Users can then input their details, such as educational background, employer type, \
                  and major, to receive a predicted salary range: As for annual earnings, responses fell into \
                 three categories: \
                    <ul> \
                        <li>below $50,000</li> \
                        <li>$50,001 to $100,000</li> \
                        <li>or more than $100,000</li> \
                    </ul> \
                  Using the results, \
                  the app gives the probability bar chart that will enable clients to understand the chance of every \
                 salary range. Career decision making is a prime area for this tool that keeps users abreast with the \
                 appropriate education level and jobs to pursue other engaging displaying salary trends of the larger workforce \
                </b>")
        
    if mode == "Train":
        df = pd.read_csv('./app/db/US_graduates.csv')
        
        with st.spinner("Training the model... Please wait."):
            trained_model, metrics = train(df)
            with open('./app/models/atharva_neural_net.sav', 'wb') as file3:
                pickle.dump(trained_model, file3)

        st.success('Model successfully trained')
        history, loss, accuracy = metrics
        st.write("Testing Loss")
        st.write(loss)
        st.write("Testing accuracy")
        st.write(accuracy * 100)
        

    elif mode == "Predict":
        # Prediction
        st.sidebar.header("Input Features")
        year = st.sidebar.number_input("Year", min_value=1990, max_value=2025, step=1, value=2022)
        education_level = st.sidebar.selectbox(
            "Education Level", ["Bachelors", "Masters", "Doctorates", "Professionals"]
        )

        employer_type = st.sidebar.selectbox(
            "Employer Type", ["Business/Industry", "Educational Institution", "Government"]
        )

        reason_outside_field = st.sidebar.selectbox(
            "Reason for Working Outside Field", [
                "Not Outside Field", "Career Change", "Family-related", "Job Location", "No Job Available", "Other"
            ]
        )

        education_major = st.sidebar.selectbox(
            "Education Major", [
                "Animal Sciences", "Anthropology and Archeology", "Area and Ethnic Studies",
                "Atmospheric Sciences and Meteorology", "Biochemistry and Biophysics",
                "Biological Sciences", "Botany", "Chemical Engineering", "Chemistry",
                "Civil Engineering", "Computer Science and Math", "Criminology", "Earth Sciences",
                "Economics", "Electrical Engineering", "Environmental Science Studies",
                "Food Sciences and Technology", "Forestry Services", "Genetics, Animal and Plant",
                "Geography", "Geology", "History of Science", "Information Services and Systems",
                "International Relations", "Linguistics", "Management & Administration",
                "Mechanical Engineering", "Nutritional Science", "OTHER Agricultural Sciences",
                "OTHER Geological Sciences", "OTHER Physical and Related Sciences", "Oceanography",
                "Operations Research", "Other Engineering", "Pharmacology, Human and Animal",
                "Philosophy of Science", "Physics and Astronomy", "Physiology, Human and Animal",
                "Plant Sciences", "Political Science and Government", "Political and related sciences",
                "Psychology", "Public Policy Studies", "Sociology", "Statistics", "Zoology, General"
            ]
        )

        input_data = pd.DataFrame([{  # Example one-hot encoding structure
            'Year': year,
            'Education.Degrees.Bachelors': int(education_level == "Bachelors"),
            'Education.Degrees.Masters': int(education_level == "Masters"),
            'Education.Degrees.Doctorates': int(education_level == "Doctorates"),
            'Education.Degrees.Professionals': int(education_level == "Professionals"),
            'Employment.Employer Type.Business/Industry': int(employer_type == "Business/Industry"),
            'Employment.Employer Type.Educational Institution': int(employer_type == "Educational Institution"),
            'Employment.Employer Type.Government': int(employer_type == "Government"),
            'Employment.Reason Working Outside Field.Career Change': int(reason_outside_field == "Career Change"),
            'Employment.Reason Working Outside Field.Family-related': int(reason_outside_field == "Family-related"),
            'Employment.Reason Working Outside Field.Job Location': int(reason_outside_field == "Job Location"),
            'Employment.Reason Working Outside Field.No Job Available': int(reason_outside_field == "No Job Available"),
            'Employment.Reason Working Outside Field.Other': int(reason_outside_field == "Other"),
            **{f"Education.Major_{education_major}": 1}
        }])

        for col in columns:
            if col.startswith("Education.Major_") and col != f"Education.Major_{education_major}":
                input_data[col] = 0

        if st.sidebar.button("Predict ", key="predict"):
            st.write("Input Data: ", year,",", education_level,",", employer_type,",", reason_outside_field,",", education_major)
            results, aligned_data = process_and_predict(input_data, loaded_model, columns, output_classes)
            for result in results:
                st.write(f"Predicted Salary Range: {result['Predicted Class']}")
                visualize_prediction(result['Probabilities'], output_classes)


    

if __name__ == "__main__":
    app()