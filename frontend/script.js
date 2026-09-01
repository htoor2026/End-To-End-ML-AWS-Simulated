const form = document.getElementById("prediction-form");

const resultCard = document.getElementById("result-card");

const predictedPrice = document.getElementById("predicted-price");

const errorBox = document.getElementById("error-box");

const button = document.getElementById("predict-button");

const buttonText = document.getElementById("button-text");

const spinner = document.getElementById("spinner");

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  resultCard.classList.add("hidden");

  errorBox.classList.add("hidden");

  button.disabled = true;

  buttonText.textContent = "Predicting...";

  spinner.classList.remove("hidden");

  const yearBuilt = Number(document.getElementById("year-built").value);

  const data = {
    Order: 1,

    PID: 5286,

    "MS SubClass": 20,

    "Lot Frontage": 80.0,

    "Lot Area": Number(document.getElementById("lot-area").value),

    "Overall Qual": Number(document.getElementById("overall-quality").value),

    "Overall Cond": Number(document.getElementById("overall-condition").value),

    "Year Built": yearBuilt,

    "Year Remod/Add": Number(document.getElementById("year-remodeled").value),

    "Mas Vnr Area": 0.0,

    "BsmtFin SF 1": 700.0,

    "BsmtFin SF 2": 0.0,

    "Bsmt Unf SF": 150.0,

    "Total Bsmt SF": 850.0,

    "1st Flr SF": 856,

    "2nd Flr SF": 854,

    "Low Qual Fin SF": 0,

    "Gr Liv Area": Number(document.getElementById("living-area").value),

    "Bsmt Full Bath": 1,

    "Bsmt Half Bath": 0,

    "Full Bath": Number(document.getElementById("full-bath").value),

    "Half Bath": 0,

    "Bedroom AbvGr": Number(document.getElementById("bedrooms").value),

    "Kitchen AbvGr": 1,

    "TotRms AbvGrd": Number(document.getElementById("rooms").value),

    Fireplaces: Number(document.getElementById("fireplaces").value),

    "Garage Yr Blt": yearBuilt,

    "Garage Cars": Number(document.getElementById("garage-cars").value),

    "Garage Area": Number(document.getElementById("garage-area").value),

    "Wood Deck SF": 210.0,

    "Open Porch SF": 0,

    "Enclosed Porch": 0,

    "3Ssn Porch": 0,

    "Screen Porch": 0,

    "Pool Area": 0,

    "Misc Val": 0,

    "Mo Sold": 5,

    "Yr Sold": 2010,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result?.detail ? JSON.stringify(result.detail) : "Prediction failed.",
      );
    }

    if (
      !result ||
      result.predicted_price === undefined ||
      result.predicted_price === null
    ) {
      throw new Error("The prediction API returned an invalid response.");
    }

    const formattedPrice = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: result.currency || "USD",

      maximumFractionDigits: 0,
    }).format(result.predicted_price);

    predictedPrice.textContent = formattedPrice;

    resultCard.classList.remove("hidden");
  } catch (error) {
    errorBox.textContent = "Unable to generate prediction: " + error.message;

    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;

    buttonText.textContent = "Predict House Price";

    spinner.classList.add("hidden");
  }
});
