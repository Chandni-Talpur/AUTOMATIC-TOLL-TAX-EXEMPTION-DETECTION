import streamlit as st
from PIL import Image
import numpy as np


def app():
    st.title("About TollTracker")

    
    # Creating two columns
    col1, col2 = st.columns([2, 0.8])  # Adjust the proportion to fit your layout

    with col1:
      st.markdown("""
      ### TollTracker: Simplifying Toll Management

      TollTracker is a comprehensive solution designed to automate the classification of vehicles and calculate toll tax dynamically. Our mission is to enhance the efficiency of toll collection systems through advanced machine learning models and user-friendly interfaces.

      **Key Features:**
      - **Automatic Vehicle Classification:** Utilizes advanced YOLO models for precise vehicle detection and classification.
      - **Tax Calculation:** Dynamically calculates tax based on vehicle type.
      - **Comprehensive Reports:** Generates daily reports for analysis and tracking.

      **Why TollTracker?**
      - **Efficiency:** Reduces manual labor and potential errors in toll collection.
      - **Accuracy:** Leverages state-of-the-art models for accurate vehicle identification.
      - **User-Friendly:** Designed with a focus on ease of use and accessibility.

      **Contact Us:**
      - **Email:** support@tolltracker.com
      - **Phone:** +123-456-7890

      Join us in revolutionizing toll collection!
      """)

    with col2:
        # Display the image with adjusted size
        image = Image.open('images/toll-booth-design-in-modern-style-easy-to-use-icon-vector.jpg')
        st.image(image, caption='Automated Toll Collection', use_column_width=True)
    

# Run the app function
if __name__ == "__main__":
    app()
  

