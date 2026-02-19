import React, { useState } from "react";
import "./App.css";

function App() {
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState("prediction will appear here");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [allPredictions, setAllPredictions] = useState({});

  // 6 activities: downstairs, jogging, sitting, standing, upstairs, walking
  const ACTIVITIES = ["downstairs", "jogging", "sitting", "standing", "upstairs", "walking"];

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const renderRow = (label, key) => (
    <div className="row">
      <div className="row-label">{label}</div>
      <input type="number" name={`${key}_x`} onChange={handleChange} />
      <input type="number" name={`${key}_y`} onChange={handleChange} />
      <input type="number" name={`${key}_z`} onChange={handleChange} />
    </div>
  );

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    
    try {
      // Validate all inputs are filled
      const requiredFields = ['att_x', 'att_y', 'att_z', 'grav_x', 'grav_y', 'grav_z', 'rot_x', 'rot_y', 'rot_z', 'acc_x', 'acc_y', 'acc_z'];
      const missingFields = requiredFields.filter(field => !inputs[field]);
      
      if (missingFields.length > 0) {
        setError("Please fill all sensor values");
        setLoading(false);
        return;
      }

      // Send request to backend
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputs),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction from backend");
      }

      const data = await response.json();
      setResult(data.activity);
      setAllPredictions(data.all_predictions || {});
    } catch (err) {
      setError("Error connecting to backend: " + err.message);
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Human Activity Recognition</h1>

      {/* Axis Header */}
      <div className="axis-header">
        <div></div>
        <div>Roll / X</div>
        <div>Yaw / Y</div>
        <div>Pitch / Z</div>
      </div>

      {renderRow("Attitude", "att")}
      {renderRow("Gravity", "grav")}
      {renderRow("Rotation", "rot")}
      {renderRow("Acceleration rate", "acc")}

      <button className="predict-btn" onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {error && <div className="error-message">{error}</div>}

      <div className="result">
        <span>Result:</span>
        <div className="result-box">{result}</div>
      </div>

      {Object.keys(allPredictions).length > 0 && (
        <div className="predictions-detail">
          <h3>Confidence Scores:</h3>
          {Object.entries(allPredictions).map(([activity, confidence]) => (
            <div key={activity} className="prediction-item">
              <span>{activity}:</span>
              <span className="confidence">{(confidence * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
