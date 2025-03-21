import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import os
import requests
# Lägg till anpassad CSS-stil för förbättrad färg och design
st.markdown("""
    <style>
        /* Huvudtitel */
        .title {
            font-size: 36px;
            color: #0a0a0a; /* Orange-röd färg */
            text-align: center;
            font-family: 'Arial', sans-serif;
        }

        /* Canvas Design */
        .stCanvas {
            border-radius: 15px;
            border: 2px solid #0a0a0a; /* Orange-röd färg för canvasen */
            background-color: #333333; /* Mörk bakgrund för canvas */
        }

        /* Kalkylatorns knappar och meny */
        .stButton, .stSelectbox, .stNumberInput {
            background-color: #0a0a0a; /* Orange-röd färg för knappar */
            color: white;
            font-size: 16px;
        }

        .stSidebar {
            background-color: #222222; /* Mörk bakgrund för sidomenyn */
        }

        /* Resultatstyling */
        .result {
            font-size: 24px;
            color: #FFD700; /* Gyllene färg för resultatet */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Försök att ladda modellen från den lokala filvägen


try:
    model_url = "https://github.com/siffror/Handwritten-Digit-Recognition-with-CNN-and-Calculator/raw/main/my_trained_model.h5"
    model_path = "my_trained_model.h5"

    # Ladda ner modellen om den inte finns lokalt (för att undvika att ladda ner varje gång)
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            response = requests.get(model_url)
            f.write(response.content)

    model = load_model(model_path)
    st.write("Modellen har laddats framgångsrikt.")
except Exception as e:
    st.write(f"Det gick inte att ladda modellen: {e}")


# Titeln på appen
st.markdown('<h1 class="title">Handskriven Sifferigenkänning med CNN och Kalkylator</h1>', unsafe_allow_html=True)
st.write("Rita en siffra i fönstret nedan och modellen kommer att förutsäga vilken siffra det är.")

# Skapa två kolumner för layout
col1, col2 = st.columns(2)

# Kolumn 1: Ritverktyg (Canvas)
with col1:
    st.write("Här kan du rita siffror.")
    canvas_result = st_canvas(
        fill_color="black",  # Fyllningsfärg
        stroke_width=20,  # Tjockleken på pennan
        stroke_color="white",  # Pennans färg
        background_color="black",  # Bakgrundsfärg
        height=280,  # Höjd på canvas
        width=280,  # Bredd på canvas
        drawing_mode="freedraw",  # Låt användaren rita fritt
        key="canvas"
    )

# Kolumn 2: Förutsägelse och kalkylator
with col2:
    predicted_label = None
    if canvas_result.image_data is not None:
        # Konvertera den ritade bilden till en PIL-bild
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        
        # Förbehandla bilden
        image = image.convert("L")  # Konvertera till gråskala
        image = image.resize((28, 28))  # Ändra storlek till 28x28
        image_array = np.array(image)  # Omvandla till array
        image_array = image_array.reshape(1, 28, 28, 1)  # Omforma till (1, 28, 28, 1)
        image_array = image_array.astype("float32") / 255.0  # Normalisera pixelvärden
        
        # Gör en förutsägelse med modellen
        try:
            prediction = model.predict(image_array)
            predicted_label = np.argmax(prediction, axis=1)[0]
            st.write(f"Modellen förutspår att detta är siffran: {predicted_label}")
        except Exception as e:
            st.write(f"Fel vid förutsägelse: {e}")

    # Om användaren har ritat en siffra, kan den användas i kalkylatorn
    if predicted_label is not None:
        st.write(f"Den förutsagda siffran från ritningen är: {predicted_label}")
        
        # Kalkylatorn i sidomenyn
        st.sidebar.header("Kalkylator")
        num2 = st.sidebar.number_input("Ange ett annat tal", value=0.0)
        operation = st.sidebar.selectbox("Välj operation", ["Addition", "Subtraktion", "Multiplikation", "Division"])
        
        # Beräkna resultatet baserat på den förutsagda siffran och användarens inmatning
        try:
            if operation == "Addition":
                result = predicted_label + num2
            elif operation == "Subtraktion":
                result = predicted_label - num2
            elif operation == "Multiplikation":
                result = predicted_label * num2
            elif operation == "Division":
                if num2 != 0:
                    result = predicted_label / num2
                else:
                    result = "Kan inte dela med 0"
            
            # Visa resultatet av kalkylatorn
            st.write(f"**Resultat**: {result}")
        except Exception as e:
            st.write(f"Det uppstod ett fel vid beräkningen: {e}")

# Förbättrad design med sidomeny och responsiv layout
st.sidebar.image("https://nyesteventuretech.com/images/Machine-Learning.jpg", use_container_width=True)
st.sidebar.write("Välkommen till handskrivna sifferigenkänning och kalkylator!")


# För att navigera till mappen där app.py finns, kör: 
# => cd C:\Users\Player1\Desktop\EC Utbildning\Machine_Learning\kunskapskontroll_2_ml_ds24
#Kör appen genom att skriva: streamlit run app.py i terminalen
