import React from "react";
export default function FilePicker({ onPick }: { onPick: (file: File) => void }) {
  return (
    <div className="card">
      <input type="file" accept="image/*" onChange={(e)=> e.target.files?.[0] && onPick(e.target.files[0])}/>
    </div>
  );
}