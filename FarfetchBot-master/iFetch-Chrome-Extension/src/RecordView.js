import { useReactMediaRecorder } from "react-media-recorder";
import { useState } from "react";

const RecordView = () => {
    
    const [isActive, setIsActive] = useState(false);
    const {
        status,
        startRecording,
        stopRecording,
        pauseRecording,
        mediaBlobUrl
      } = useReactMediaRecorder({
        video: false,
        audio: true,
        echoCancellation: true
      });
      console.log("url", mediaBlobUrl);

  return (
    <div>
      
    </div>
  );
};

export default RecordView;