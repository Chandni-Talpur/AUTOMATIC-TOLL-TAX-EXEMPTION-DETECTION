import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
import onnxruntime as ort
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('your credentials')
    firebase_admin.initialize_app(cred)

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host='host name',
        user='user name',
        password='password',
        database='database name'
    )

def app():
    st.title('Welcome to Toll Tax Tracker')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''

    def login():
        email = st.session_state.login_email
        password = st.session_state.login_password
        try:
            user = auth.get_user_by_email(email)
            st.write('Login successful!')
            st.session_state.username = user.uid
            st.session_state.useremail = user.email
            st.session_state.signout = True
            st.session_state.signedout = True
        except:
            st.warning('Login Failed')

    def signout():
        st.session_state.signout = False
        st.session_state.signedout = False
        st.session_state.username = ''
        st.session_state.useremail = ''

    if "signedout" not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False

    if not st.session_state['signedout']:
        choice = st.selectbox('Login/Sign Up', ['Login', 'Sign Up'])

        if choice == 'Login':
            st.text_input('Email Address', key='login_email')
            st.text_input('Password', type='password', key='login_password')
            st.button('Login', on_click=login)

        else:
            email = st.text_input('Email Address')
            password = st.text_input('Password', type='password')
            username = st.text_input('Enter your unique username')

            if st.button('Create my account'):
                try:
                    user = auth.create_user(email=email, password=password, uid=username)
                    st.success('Account created successfully!')
                    st.markdown('Please login using your email and password')
                    st.balloons()
                except Exception as e:
                    st.warning(f'Failed to create account: {e}')

    if st.session_state.signout:

        # home code start

        # Initialize session state variables if not already present
        if 'total_no_of_vehicles' not in st.session_state:
            st.session_state.total_no_of_vehicles = 0
        if 'no_of_vehicles_exempted' not in st.session_state:
            st.session_state.no_of_vehicles_exempted = 0
        if 'no_of_vehicle_paidtax' not in st.session_state:
            st.session_state.no_of_vehicle_paidtax = 0
        if 'totaltax_collected' not in st.session_state:
            st.session_state.totaltax_collected = 0

        # Load models
        classification_model = ort.InferenceSession("classification.onnx")
        detection_model = YOLO('object_detection.pt')

        # Tax rates
        tax_rates = {
            'ambulance': 0,
            'bolan': 70,
            'bus': 130,
            'car': 40,
            'police': 0,
            'suzuki': 70,
            'trailor': 350,
            'truck': 150,
            'van': 70
        }

        def classify_vehicle(image):
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)

            inputs = {classification_model.get_inputs()[0].name: image}
            outputs = classification_model.run(None, inputs)
            output = np.array(outputs[0])
            classes = ['ambulance', 'bolan', 'bus', 'car', 'police', 'suzuki', 'trailor', 'truck',      'van']
            return classes[np.argmax(output)]

        def detect_objects(image):
            results = detection_model(image)    
            return results

        def extract_labels(results):
            labels = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    labels.append(label)
            return labels

        def calculate_tax(vehicle_type, detection_labels):
            total_no_of_vehicles = 1
            no_of_vehicles_exempted = 0
            no_of_vehicle_paidtax = 0
            totaltax_collected = 0
            if 'MUET' in detection_labels or vehicle_type in ['ambulance', 'police']:
                tax = 0
                no_of_vehicles_exempted += 1
            else:
                tax = tax_rates[vehicle_type]
                no_of_vehicle_paidtax += 1
            totaltax_collected += tax
            return total_no_of_vehicles, no_of_vehicles_exempted, no_of_vehicle_paidtax,     totaltax_collected
    
        st.markdown(
            """
            <style>
            body {
                background-color: #f0f2f6; /* Light grey */
                color: #333; /* Default text color */
            }
            .stButton>button {
                background-color: #4CAF50; /* Green */
                color: white;
                border-radius: 12px;
                padding: 10px 20px;
                border: none;
            }        
            .stUpload>label {
                color: #333; /* Darker grey */
                font-weight: bold;
            }
            .reportview-container .main .block-container {
                padding-top: 50px; /* Top padding */
            }
            </style>
             """,
             unsafe_allow_html=True
        )

        # st.title("Vehicle Classification and Tax Calculation")

        st.markdown('<h1 style="font-size: 28px; color: #D3E9D2;">Vehicle Classification and Tax Calculation</h1>', unsafe_allow_html=True)

        st.markdown(
            "<style>.stUpload > label { font-size: 20px; }</style>", 
            unsafe_allow_html=True
        )

    
        # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

        # Option to go live or upload image
        option = st.selectbox('Choose input method', ['Upload Image', 'Go Live'])

        # Handle image upload
        if option == 'Upload Image':
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                vehicle_type = classify_vehicle(image)
                detection_results = detect_objects(image)
                detection_labels = extract_labels(detection_results)

                total_no_of_vehicles, no_of_vehicles_exempted, no_of_vehicle_paidtax,         totaltax_collected = calculate_tax(vehicle_type, detection_labels)

                st.session_state.total_no_of_vehicles += total_no_of_vehicles
                st.session_state.no_of_vehicles_exempted += no_of_vehicles_exempted
                st.session_state.no_of_vehicle_paidtax += no_of_vehicle_paidtax
                st.session_state.totaltax_collected += totaltax_collected

                def get_db_connection():
                    return mysql.connector.connect(
                        host='host name',
                        user='user name',
                        password='password',
                        database='database name'
                    )
        
                if 'MUET' in detection_labels:
                    tax = 0
                    exemption_reason = 'MUET'
                elif vehicle_type in ['ambulance', 'police']:
                    tax = 0
                    exemption_reason = 'Public Service Vehicle'
                else:
                    tax = tax_rates[vehicle_type]
                    exemption_reason = '_'
        
                def insert_data(vehicle_type, tax, total_no_of_vehicles, no_of_vehicles_exempted,         no_of_vehicle_paidtax, totaltax_collected):
                    connection = get_db_connection()
                    cursor1 = connection.cursor()

                # Get the current date and time in 24-hour format
                    current_date = datetime.now().date()
                    current_time = datetime.now().strftime("%H:%M:%S")  # 24-hour format
                    query1 = """
                    INSERT INTO dailyreport (date, time, vehicle_type, tax, exemption_reason,     user_email)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor1.execute(query1, (current_date, current_time, vehicle_type, tax,     exemption_reason, st.session_state.useremail))
                    connection.commit()
                    cursor1.close()
            
                    cursor2 = connection.cursor()
                    query2 = """
                    INSERT INTO DayWiseReport (date, total_no_of_vehicles, no_of_vehicle_paidtax,         no_of_vehicle_exemptedtax, total_taxcollected, user_email)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                    cursor2.execute(query2, (current_date, total_no_of_vehicles, no_of_vehicle_paidtax,    no_of_vehicles_exempted, totaltax_collected, st.session_state.useremail))
                    connection.commit()
                    cursor2.close()

                    connection.close()

                insert_data(vehicle_type, tax, total_no_of_vehicles, no_of_vehicles_exempted,         no_of_vehicle_paidtax, totaltax_collected)

                st.image(image, caption='Uploaded Image.', use_column_width=True)
                st.write(f"Vehicle type: {vehicle_type}")
                st.write(f"Tax: {tax}")

                st.markdown('<h1 style="font-size: 28px; color: green;">Tax Collection Report</    h1>',     unsafe_allow_html=True)

                st.write(f"Total Vehicles: {st.session_state.total_no_of_vehicles}")
                st.write(f"Total Tax Collected: {st.session_state.totaltax_collected}")
                st.write(f"Vehicles that paid tax: {st.session_state.no_of_vehicle_paidtax}")
                st.write(f"Vehicles exempted from tax: {st.session_state.no_of_vehicles_exempted}")
        

        # Handle live camera input
        elif option == 'Go Live':
            # Function to display live video and automatically capture frame
            def live_stream(image_count):
                cap = cv2.VideoCapture(0)  # Open default camera
                frame_count = 0  # Counter for frames
                captured_frame = None  # Variable to store the frame

                stframe = st.empty()  # Placeholder for video feed

                while frame_count < 100:
                    ret, frame = cap.read()  # Read frame-by-frame
                    if not ret:
                        st.error("Failed to capture video")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB")

                    frame_count += 1  # Increment frame count

                    # Save the 5th frame and stop the stream
                    if frame_count == 100:
                        captured_frame = frame  # Save the frame in BGR format for saving with OpenCV
                        break

                cap.release()  # Release the video capture object

                if captured_frame is not None:
                    # Save the frame with a sequential filename
                    file_name = f'captured_frame_{image_count}.jpg'
                    cv2.imwrite(file_name, captured_frame)  # Save the frame in BGR format
                    return file_name
                return None

            # Main Streamlit app
            # st.title("Live Video Stream - Capture and Save Frames")
            st.markdown('<h1 style="font-size: 24px; color: #D3E9D2;">Live Video Stream - Capture and Save Frames</h1>', unsafe_allow_html=True)

            # Initialize or load the image counter from session state
            if 'image_count' not in st.session_state:
                st.session_state.image_count = 1  # Start image count at 1

            if st.button("Start Live Stream"):
                file_name = live_stream(st.session_state.image_count)  # Capture and save frame

                if file_name:
                    # Increment the image count after saving
                    st.session_state.image_count += 1

                    # Display a success message and show the saved frame
                    st.success(f"frame captured and saved as {file_name}.")
        
                    # Load the saved image and display it in Streamlit
                    img = Image.open(file_name)
                    st.image(img, caption=f"Captured Frame {st.session_state.image_count - 1}",             use_column_width=True)

                    uploaded_file = file_name

                    if uploaded_file is not None:

                        with open(uploaded_file, 'rb') as img_file:
                            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                            # Decode the image for processing using OpenCV
                            image = cv2.imdecode(file_bytes, 1)

                        vehicle_type = classify_vehicle(image)
                        detection_results = detect_objects(image)
                        detection_labels = extract_labels(detection_results)

                        total_no_of_vehicles, no_of_vehicles_exempted, no_of_vehicle_paidtax,                 totaltax_collected = calculate_tax(vehicle_type, detection_labels)

                        st.session_state.total_no_of_vehicles += total_no_of_vehicles
                        st.session_state.no_of_vehicles_exempted += no_of_vehicles_exempted
                        st.session_state.no_of_vehicle_paidtax += no_of_vehicle_paidtax
                        st.session_state.totaltax_collected += totaltax_collected

                        def get_db_connection():
                            return mysql.connector.connect(
                                host='host name',
                                user='user name',
                                password='password',
                                database='database name'
                            )
        
                        if 'MUET' in detection_labels:
                            tax = 0
                            exemption_reason = 'MUET'
                        elif vehicle_type in ['ambulance', 'police']:
                            tax = 0
                            exemption_reason = 'Public Service Vehicle'
                        else:
                            tax = tax_rates[vehicle_type]
                            exemption_reason = '_'
        
                        def insert_data(vehicle_type, tax, total_no_of_vehicles,         no_of_vehicles_exempted,         no_of_vehicle_paidtax, totaltax_collected):
                            connection = get_db_connection()
                            cursor1 = connection.cursor()

                        # Get the current date and time in 24-hour format
                            current_date = datetime.now().date()
                            current_time = datetime.now().strftime("%H:%M:%S")  # 24-hour format
                            query1 = """
                            INSERT INTO dailyreport (date, time, vehicle_type, tax, exemption_reason,             user_email)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """
                            cursor1.execute(query1, (current_date, current_time, vehicle_type, tax,             exemption_reason, st.session_state.useremail))
                            connection.commit()
                            cursor1.close()
            
                            cursor2 = connection.cursor()
                            query2 = """
                            INSERT INTO DayWiseReport (date, total_no_of_vehicles,         no_of_vehicle_paidtax,         no_of_vehicle_exemptedtax, total_taxcollected,         user_email)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                            cursor2.execute(query2, (current_date, total_no_of_vehicles,         no_of_vehicle_paidtax,    no_of_vehicles_exempted, totaltax_collected, st.        session_state.useremail))
                            connection.commit()
                            cursor2.close()

                            connection.close()

                        insert_data(vehicle_type, tax, total_no_of_vehicles,         no_of_vehicles_exempted,         no_of_vehicle_paidtax, totaltax_collected)

                        st.image(image, caption='Uploaded Image.', use_column_width=True)
                        st.write(f"Vehicle type: {vehicle_type}")
                        st.write(f"Tax: {tax}")

                        st.markdown('<h1 style="font-size: 28px; color: green;">Tax Collection Report</            h1>',     unsafe_allow_html=True)

                        st.write(f"Total Vehicles: {st.session_state.total_no_of_vehicles}")
                        st.write(f"Total Tax Collected: {st.session_state.totaltax_collected}")
                        st.write(f"Vehicles that paid tax: {st.session_state.no_of_vehicle_paidtax}")
                        st.write(f"Vehicles exempted from tax: {st.session_state.no_of_vehicles_exempted}")
        

                else:
                    st.warning("Failed to capture the 5th frame.")



        # home code ending

        # Database connection
        def get_db_connection():
            return mysql.connector.connect(
                host='host name',
                user='user name',
                password='password',
                database='database name'
            )

        st.markdown('<h1 style="font-size: 28px; color: #D3E9D2;">Daily report</h1>', unsafe_allow_html=True)

        connection = get_db_connection()

        cursor1 = connection.cursor()

        # cursor1.execute("SELECT * FROM dailyreport")
        cursor1.execute(f"SELECT date, time, vehicle_type, tax, exemption_reason FROM dailyreport WHERE user_email = '{st.session_state.useremail}'")
        data1 = cursor1.fetchall()

        df1 = pd.DataFrame(data1, columns=cursor1.column_names)

       # Check if the 'time' column exists and convert it to 12-hour format with AM/PM
        if 'time' in df1.columns:  # Replace 'time' with your actual column name
            def convert_to_12hr_format(x):
                if isinstance(x, timedelta):
                    return (datetime.min + x).time().strftime('%I:%M:%S %p')
                return x

            df1['time'] = df1['time'].apply(convert_to_12hr_format)


        st.dataframe(df1)

        cursor1.close()

        st.markdown('<h1 style="font-size: 28px; color: #D3E9D2;">Day wise report</h1>', unsafe_allow_html=True)

        cursor2 = connection.cursor()

        cursor2.execute(f"""
            SELECT 
                date,
                SUM(total_no_of_vehicles) AS total_no_of_vehicles,
                SUM(no_of_vehicle_paidtax) AS no_of_vehicle_paidtax,
                SUM(no_of_vehicle_exemptedtax) AS no_of_vehicle_exemptedtax,
                SUM(total_taxcollected) AS total_taxcollected
            FROM 
                DayWiseReport
            WHERE 
                user_email = '{st.session_state.useremail}'
            GROUP BY 
                date
             ORDER BY 
                date;
        """)
        
        data2 = cursor2.fetchall()

        df2 = pd.DataFrame(data2, columns=['date', 'total_no_of_vehicles', 'no_of_vehicle_paidtax', 'no_of_vehicle_exemptedtax', 'total_taxcollected'])
        st.dataframe(df2)

        cursor2.close()

        connection.close()

        st.button('Sign out', on_click=signout)

if __name__ == '__main__':
    app()
