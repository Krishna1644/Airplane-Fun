import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import Papa from "papaparse";

const API_BASE_URL = "http://localhost:8000"; // Make sure the FastAPI backend is running!

function App() {
  const [formData, setFormData] = useState({
    Origin_Airport: "ATL",
    Destination_Airport: "LAX",
    Carrier: "Delta Air Lines Inc",
    Flight_Time: "12:00",
    Date: new Date().toISOString().split("T")[0], // Defaults to today
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [airports, setAirports] = useState([]);

  // Load airport data for the map
  useEffect(() => {
    fetch("/airport_performance_tiers_enriched.csv")
      .then((response) => response.text())
      .then((csvData) => {
        Papa.parse(csvData, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const validAirports = results.data.filter(
              (a) => a.LATITUDE && a.LONGITUDE,
            );
            setAirports(validAirports);
          },
        });
      });
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    try {
      // Calculate Time Block based on selected Flight Time (e.g. 15:30)
      let timeBlock = "Unknown";
      if (formData.Flight_Time) {
        const [hours, minutes] = formData.Flight_Time.split(":");
        const timeNum = parseInt(hours, 10) * 100 + parseInt(minutes, 10);

        if (timeNum >= 500 && timeNum < 1200) timeBlock = "Morning";
        else if (timeNum >= 1200 && timeNum < 1700) timeBlock = "Afternoon";
        else if (timeNum >= 1700 && timeNum < 2100) timeBlock = "Evening";
        else timeBlock = "Night";
      }

      const payload = {
        ...formData,
        Departure_Time: timeBlock,
      };

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("API Request Failed");
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
      alert(
        "Error reaching the backend API. Make sure FastAPI is running on port 8000.",
      );
    } finally {
      setLoading(false);
    }
  };

  const getTierColor = (tier) => {
    const colors = {
      0: "#3b82f6", // Secondary
      1: "#ef4444", // High Risk
      2: "#f59e0b", // Underperforming
      3: "#10b981", // Efficient
      4: "#8b5cf6", // MegaHub
    };
    return colors[tier] || "#64748b";
  };

  const getRiskClass = (prob) => {
    if (prob > 0.5) return "risk-high";
    if (prob > 0.2) return "risk-med";
    return "risk-low";
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>✈️ Flight Risk Evaluator</h1>
        <div style={{ fontSize: "0.85rem", color: "#64748b" }}>
          Powered by AI & Live Weather
        </div>
      </header>

      <main className="main-content">
        <aside className="glass-panel">
          <form onSubmit={handlePredict}>
            <h2>Trip Details</h2>

            <label>Origin Airport (IATA Code)</label>
            <input
              name="Origin_Airport"
              value={formData.Origin_Airport}
              onChange={handleInputChange}
              placeholder="e.g. ATL, ORD"
              maxLength={3}
              required
            />

            <label>Destination Airport (IATA Code)</label>
            <input
              name="Destination_Airport"
              value={formData.Destination_Airport}
              onChange={handleInputChange}
              placeholder="e.g. LAX, JFK"
              maxLength={3}
              required
            />

            <label>Carrier / Airline</label>
            <select
              name="Carrier"
              value={formData.Carrier}
              onChange={handleInputChange}
            >
              <option value="Delta Air Lines Inc">Delta Air Lines Inc</option>
              <option value="American Airlines Inc.">
                American Airlines Inc.
              </option>
              <option value="United Air Lines Inc.">
                United Air Lines Inc.
              </option>
              <option value="Southwest Airlines Co.">
                Southwest Airlines Co.
              </option>
              <option value="Alaska Airlines Inc.">Alaska Airlines Inc.</option>
              <option value="JetBlue Airways">JetBlue Airways</option>
              <option value="Spirit Air Lines">Spirit Air Lines</option>
            </select>

            <label>Flight Date (Format: YYYY-MM-DD)</label>
            <input
              type="date"
              name="Date"
              value={formData.Date}
              onChange={handleInputChange}
              required
            />

            <label>Flight Time</label>
            <input
              type="time"
              name="Flight_Time"
              value={formData.Flight_Time}
              onChange={handleInputChange}
              required
            />

            <button type="submit" disabled={loading}>
              {loading
                ? "Analyzing Trends & Weather..."
                : "Evaluate Flight Risk"}
            </button>
          </form>

          {prediction && (
            <div className="results-panel">
              <div className="result-card">
                <div className="result-label">Expected Departure Delay</div>
                <div className="result-value">
                  {prediction.expected_dep_delay > 0 ? `+${prediction.expected_dep_delay}` : '0'} m
                </div>
                <div style={{fontSize: '0.75rem', color: '#64748b'}}>
                  Base Environmental Risk (Weather, Traffic, Ops)
                </div>
              </div>

              <div className="result-card">
                <div className="result-label">Expected Arrival Delay</div>
                <div className="result-value">
                  {prediction.expected_arr_delay > 0 ? `+${prediction.expected_arr_delay}` : '0'} m
                </div>
                <div style={{fontSize: '0.75rem', color: '#64748b'}}>
                  Total Trip Prediction (including in-air recovery)
                </div>
              </div>
              
              <div className="result-card">
                <div className="result-label">Severe Risk Index</div>
                <div className={`result-value ${getRiskClass(prediction.risk_index / 100)}`}>
                  {prediction.risk_index.toFixed(1)} / 100
                </div>
                <div style={{fontSize: '0.75rem', color: '#64748b'}}>
                  Apriori blended Confidence & Lift score
                </div>
              </div>

              <div className="matched-rules">
                <strong>Local Weather Forecast for {formData.Date}:</strong><br/>
                🌡️ {prediction.weather_used.avg_temp_f.toFixed(1)}°F Avg Temp<br/>
                🌧️ {prediction.weather_used.precipitation_inches.toFixed(2)} in. Rain, ❄️ {prediction.weather_used.snowfall_inches.toFixed(2)} in. Snow<br/>
                💨 {prediction.weather_used.wind_speed_mph.toFixed(1)} mph Wind
              </div>

              {prediction.matched_conditions && prediction.matched_conditions[0] !== "Baseline Risk" && (
                 <div className="matched-rules" style={{marginTop: '0.5rem', borderColor: 'orange', borderStyle: 'solid', borderWidth: '1px'}}>
                   <strong>Triggering Rule Conditions:</strong> <br/>
                   <span style={{color: '#94a3b8'}}>{prediction.matched_conditions.join(' + ')}</span>
                 </div>
              )}
            </div>
          )}
        </aside>

        <section className="map-container">
          <MapContainer
            center={[39.8283, -98.5795]}
            zoom={4}
            className="leaflet-container"
            zoomControl={false}
          >
            {/* Dark mode carto map tiles */}
            <TileLayer
              attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />
            {airports.map((airport, idx) => (
              <CircleMarker
                key={idx}
                center={[airport.LATITUDE, airport.LONGITUDE]}
                radius={airport.Delay_Class_Severe > 0.2 ? 6 : 4}
                fillColor={getTierColor(airport.Performance_Tier)}
                color={getTierColor(airport.Performance_Tier)}
                weight={1}
                opacity={0.8}
                fillOpacity={0.6}
                eventHandlers={{
                  click: () => {
                    setFormData((prev) => ({
                      ...prev,
                      Origin_Airport: airport.Dep_Airport,
                    }));
                  },
                }}
              >
                <Popup>
                  <div style={{ color: "#1e293b", padding: "5px" }}>
                    <strong>
                      {airport.Dep_Airport} - {airport.AIRPORT}
                    </strong>
                    <br />
                    Tier: {airport.Performance_Tier}
                    <br />
                    Avg Delay: {airport.avg_dep_delay?.toFixed(1)} min
                    <br />
                    <i>Click marker to select as Origin</i>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
            
            {/* Draw flight path if prediction is available */}
            {prediction && prediction.flight_path && (
              <>
                {/* Glow effect */}
                <Polyline 
                  positions={prediction.flight_path} 
                  color="#38bdf8" 
                  weight={6} 
                  opacity={0.3} 
                />
                {/* Animated dash line */}
                <Polyline 
                  positions={prediction.flight_path} 
                  color="#ffffff" 
                  weight={2.5} 
                  className="animated-flight-path"
                  opacity={1} 
                />
              </>
            )}
          </MapContainer>
        </section>
      </main>
    </div>
  );
}

export default App;
