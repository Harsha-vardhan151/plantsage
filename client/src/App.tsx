// client/src/App.tsx  (top navbar + routes)
import React from "react";
import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import IdentifyPage from "./pages/IdentifyPage";
import DiseasePage from "./pages/DiseasePage";

export default function App() {
  return (
    <div className="page">
      <header className="app-header">
        <h1>🌿 PlantSage</h1>
        <nav className="tabs">
          <NavLink to="/identify" className={({isActive})=> isActive ? "tab active" : "tab"}>Identification</NavLink>
          <NavLink to="/disease"  className={({isActive})=> isActive ? "tab active" : "tab"}>Disease detection</NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/identify" replace />} />
        <Route path="/identify" element={<IdentifyPage />} />
        <Route path="/disease"  element={<DiseasePage  />} />
        <Route path="*" element={<Navigate to="/identify" replace />} />
      </Routes>

      <footer className="muted small">
        Images stay on your server. Predictions may be wrong—verify before handling plants.
      </footer>
    </div>
  );
}
