const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

const form = document.getElementById("prediction-form");
const resultCard = document.getElementById("result-card");
const predictedPrice = document.getElementById("predicted-price");
const errorBox = document.getElementById("error-box");
const button = document.getElementById("predict-button");
const buttonText = document.getElementById("button-text");
const spinner = document.getElementById("spinner");

const heroStatusDot = document.getElementById("hero-status-dot");
const heroStatusText = document.getElementById("hero-status-text");

let technicalStatusLoaded = false;


function activateTab(tabName) {
  tabButtons.forEach((tabButton) => {
    const isActive = tabButton.dataset.tab === tabName;

    tabButton.classList.toggle("active", isActive);
    tabButton.setAttribute(
      "aria-selected",
      isActive ? "true" : "false",
    );
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.id === `${tabName}-tab`;

    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });

  if (tabName === "technical" && !technicalStatusLoaded) {
    loadTechnicalOverview();
  }
}


tabButtons.forEach((tabButton) => {
  tabButton.addEventListener("click", () => {
    activateTab(tabButton.dataset.tab);
  });
});


function prettifyStatus(value) {
  if (!value) {
    return "Unavailable";
  }

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}


function setStatus(elementId, value, state = "neutral") {
  const element = document.getElementById(elementId);

  if (!element) {
    return;
  }

  element.textContent = value;
  element.dataset.state = state;
}


async function fetchJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }

  return response.json();
}


async function loadApiHealth() {
  try {
    const health = await fetchJson("/health");

    if (health.status === "healthy") {
      heroStatusDot.dataset.state = "healthy";
      heroStatusText.textContent = "API online";
      setStatus("api-health", "Healthy", "healthy");
      setStatus("api-health-detail", "FastAPI is responding");
      return;
    }

    throw new Error("Unexpected health response");
  } catch (error) {
    heroStatusDot.dataset.state = "error";
    heroStatusText.textContent = "API status unavailable";
    setStatus("api-health", "Unavailable", "error");
    setStatus("api-health-detail", error.message);
  }
}


async function loadTechnicalOverview() {
  technicalStatusLoaded = true;

  await loadApiHealth();

  try {
    const drift = await fetchJson("/drift");
    const driftStatus = drift.overall_status;
    const state =
      driftStatus === "healthy"
        ? "healthy"
        : driftStatus === "warning"
          ? "warning"
          : driftStatus === "critical"
            ? "error"
            : "neutral";

    setStatus(
      "drift-health",
      prettifyStatus(driftStatus),
      state,
    );

    const sampleCounts = Array.isArray(drift.rows)
      ? drift.rows
          .map((row) => Number(row.production_samples || 0))
          .filter((value) => Number.isFinite(value))
      : [];

    const latestSamples =
      sampleCounts.length > 0 ? Math.max(...sampleCounts) : 0;

    setStatus(
      "drift-health-detail",
      latestSamples > 0
        ? `${latestSamples} recent prediction records evaluated`
        : "Waiting for enough production predictions",
    );
  } catch (error) {
    setStatus("drift-health", "Unavailable", "error");
    setStatus("drift-health-detail", error.message);
  }

  try {
    const performance = await fetchJson("/performance");
    const performanceStatus = performance.status;
    const state =
      performanceStatus === "healthy"
        ? "healthy"
        : performanceStatus === "degraded"
          ? "error"
          : "neutral";

    setStatus(
      "performance-health",
      prettifyStatus(performanceStatus),
      state,
    );

    if (performance.mae !== null && performance.mae !== undefined) {
      const formattedMae = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(performance.mae);

      setStatus(
        "performance-health-detail",
        `${performance.samples} labels · MAE ${formattedMae}`,
      );
    } else {
      setStatus(
        "performance-health-detail",
        `${performance.samples || 0} labelled outcomes available`,
      );
    }
  } catch (error) {
    setStatus("performance-health", "Unavailable", "error");
    setStatus("performance-health-detail", error.message);
  }
}


form.addEventListener("submit", async function (event) {
  event.preventDefault();

  resultCard.classList.add("hidden");
  errorBox.classList.add("hidden");

  button.disabled = true;
  buttonText.textContent = "Predicting...";
  spinner.classList.remove("hidden");

  const yearBuilt = Number(
    document.getElementById("year-built").value,
  );

  const data = {
    Order: 1,
    PID: 5286,
    "MS SubClass": 20,
    "Lot Frontage": 80.0,
    "Lot Area": Number(
      document.getElementById("lot-area").value,
    ),
    "Overall Qual": Number(
      document.getElementById("overall-quality").value,
    ),
    "Overall Cond": Number(
      document.getElementById("overall-condition").value,
    ),
    "Year Built": yearBuilt,
    "Year Remod/Add": Number(
      document.getElementById("year-remodeled").value,
    ),
    "Mas Vnr Area": 0.0,
    "BsmtFin SF 1": 700.0,
    "BsmtFin SF 2": 0.0,
    "Bsmt Unf SF": 150.0,
    "Total Bsmt SF": 850.0,
    "1st Flr SF": 856,
    "2nd Flr SF": 854,
    "Low Qual Fin SF": 0,
    "Gr Liv Area": Number(
      document.getElementById("living-area").value,
    ),
    "Bsmt Full Bath": 1,
    "Bsmt Half Bath": 0,
    "Full Bath": Number(
      document.getElementById("full-bath").value,
    ),
    "Half Bath": 0,
    "Bedroom AbvGr": Number(
      document.getElementById("bedrooms").value,
    ),
    "Kitchen AbvGr": 1,
    "TotRms AbvGrd": Number(
      document.getElementById("rooms").value,
    ),
    Fireplaces: Number(
      document.getElementById("fireplaces").value,
    ),
    "Garage Yr Blt": yearBuilt,
    "Garage Cars": Number(
      document.getElementById("garage-cars").value,
    ),
    "Garage Area": Number(
      document.getElementById("garage-area").value,
    ),
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
        result?.detail
          ? JSON.stringify(result.detail)
          : "Prediction failed.",
      );
    }

    if (
      !result ||
      result.predicted_price === undefined ||
      result.predicted_price === null
    ) {
      throw new Error(
        "The prediction API returned an invalid response.",
      );
    }

    const formattedPrice = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: result.currency || "USD",
      maximumFractionDigits: 0,
    }).format(result.predicted_price);

    predictedPrice.textContent = formattedPrice;
    resultCard.classList.remove("hidden");
  } catch (error) {
    errorBox.textContent =
      "Unable to generate prediction: " + error.message;

    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
    buttonText.textContent = "Predict House Price";
    spinner.classList.add("hidden");
  }
});


loadApiHealth();
