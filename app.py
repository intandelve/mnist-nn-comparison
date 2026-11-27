
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image
from streamlit_drawable_canvas import st_canvas

from models.softmax_regression import SoftmaxRegression
from models.neural_network import NeuralNetwork


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

st.title("Handwritten Digit Recognition")

st.write(
    "Draw a handwritten digit from 0 to 9 and compare "
    "the predictions of Softmax Regression and Neural Network (MLP)."
)


# ==========================================
# 2. SESSION STATE
# ==========================================

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0


def clear_canvas():
    st.session_state.canvas_version += 1
    st.session_state.pop("prediction_results", None)
    st.session_state.pop("processed_image", None)


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    softmax_model = SoftmaxRegression()
    mlp_model = NeuralNetwork()

    softmax_model.load_state_dict(
        torch.load(
            "results/softmax.pth",
            map_location="cpu"
        )
    )

    mlp_model.load_state_dict(
        torch.load(
            "results/neural_network.pth",
            map_location="cpu"
        )
    )

    softmax_model.eval()
    mlp_model.eval()

    return softmax_model, mlp_model


softmax_model, mlp_model = load_models()


# ==========================================
# 4. PREPROCESS DRAWING
# ==========================================

def preprocess_drawing(image_data):

    # Convert canvas data to PIL image
    image = Image.fromarray(
        image_data.astype("uint8"),
        "RGBA"
    )

    # Convert to grayscale
    image = image.convert("L")

    # Canvas uses white strokes on black background
    # Keep the same foreground/background convention as MNIST

    # Find the digit boundaries
    binary_image = image.point(
        lambda p: 255 if p > 30 else 0
    )

    bbox = binary_image.getbbox()

    if bbox is None:
        return None, None

    # Crop the digit
    image = image.crop(bbox)

    # Resize while preserving aspect ratio
    image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create a 28x28 black image
    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center the digit
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Resize to 14x14
    canvas = canvas.resize(
        (14, 14),
        Image.Resampling.LANCZOS
    )

    # Normalize pixel values
    array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # Flatten into 196 features
    tensor = torch.tensor(
        array,
        dtype=torch.float32
    ).reshape(1, 196)

    return tensor, canvas


# ==========================================
# 5. DRAWING CANVAS
# ==========================================

st.subheader("Draw Your Digit")

col_canvas, col_info = st.columns([1, 1])

with col_canvas:

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"digit_canvas_{st.session_state.canvas_version}"
    )

with col_info:

    st.markdown("### Instructions")

    st.write("1. Draw one digit from 0 to 9.")
    st.write("2. Click Predict.")
    st.write("3. Compare both model predictions.")

    st.info(
        "The models are trained using the MNIST dataset. "
        "They always select one digit from 0 to 9."
    )


# ==========================================
# 6. BUTTONS
# ==========================================

col_predict, col_clear = st.columns(2)

with col_predict:

    predict_button = st.button(
        "Predict",
        type="primary",
        use_container_width=True
    )

with col_clear:

    st.button(
        "Clear Drawing",
        on_click=clear_canvas,
        use_container_width=True
    )


# ==========================================
# 7. PREDICTION
# ==========================================

if predict_button:

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        input_tensor, processed_image = preprocess_drawing(
            canvas_result.image_data
        )

        if input_tensor is None:

            st.warning("Please draw a digit first.")

        else:

            with torch.no_grad():

                # Softmax Regression
                softmax_output = softmax_model(input_tensor)

                softmax_probs = F.softmax(
                    softmax_output,
                    dim=1
                )[0]

                # Neural Network (MLP)
                mlp_output = mlp_model(input_tensor)

                mlp_probs = F.softmax(
                    mlp_output,
                    dim=1
                )[0]

                # Predicted digits
                softmax_digit = torch.argmax(
                    softmax_probs
                ).item()

                mlp_digit = torch.argmax(
                    mlp_probs
                ).item()

            # Save prediction results
            st.session_state["prediction_results"] = {
                "softmax_digit": softmax_digit,
                "mlp_digit": mlp_digit,
                "softmax_probs": softmax_probs.numpy(),
                "mlp_probs": mlp_probs.numpy()
            }

            st.session_state["processed_image"] = processed_image


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

if "prediction_results" in st.session_state:

    results = st.session_state["prediction_results"]

    st.divider()

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Softmax Regression")

        st.metric(
            "Predicted Digit",
            results["softmax_digit"]
        )

        st.write(
            f"Confidence: "
            f"{results['softmax_probs'][results['softmax_digit']]:.2%}"
        )

    with col2:

        st.markdown("### Neural Network (MLP)")

        st.metric(
            "Predicted Digit",
            results["mlp_digit"]
        )

        st.write(
            f"Confidence: "
            f"{results['mlp_probs'][results['mlp_digit']]:.2%}"
        )


    # ======================================
    # 9. PROCESSED INPUT IMAGE
    # ======================================

    st.subheader("Processed Input Image")

    st.write(
        "This enlarged image shows the 14x14 input "
        "actually received by the models."
    )

    enlarged_image = st.session_state[
        "processed_image"
    ].resize(
        (280, 280),
        Image.Resampling.NEAREST
    )

    st.image(
        enlarged_image,
        caption="Enlarged 14x14 model input",
        width=280
    )


    # ======================================
    # 10. MODEL COMPARISON
    # ======================================

    st.subheader("Model Comparison")

    if results["softmax_digit"] == results["mlp_digit"]:

        st.success(
            f"Both models predict digit "
            f"{results['mlp_digit']}."
        )

    else:

        st.info(
            "The two models produce different predictions."
        )


    # ======================================
    # 11. CONFIDENCE CHART
    # ======================================

    st.subheader("Prediction Confidence by Digit")

    confidence_df = pd.DataFrame(
        {
            "Softmax Regression": results["softmax_probs"],
            "Neural Network (MLP)": results["mlp_probs"]
        },
        index=list(range(10))
    )

    confidence_df.index.name = "Digit"

    st.bar_chart(
        confidence_df,
        y=["Softmax Regression", "Neural Network (MLP)"],
        y_label="Probability",
        x_label="Digit",
        height=350
    )

    st.caption(
        "Confidence represents the predicted probability, "
        "not a guarantee that the prediction is correct."
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "MNIST Handwritten Digit Recognition | "
    "Softmax Regression vs. Neural Network (MLP)"
)
