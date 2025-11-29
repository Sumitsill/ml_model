# FastAPI imports
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Core libraries
import cv2
import json
import os
import sys
from pathlib import Path
import uuid
from datetime import datetime
import io

# directory add
sys.path.append(str(Path(__file__).parent.parent))

from utils import PoseCalibrator
from metrics import PerformanceMetrics
import time

# gher gher gher FastAPI
app = FastAPI()

#  CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# g storge fro vdo processing
video_results = {}

class ExerciseSession:
    """Handles exercise analysis for a single video session"""
    def __init__(self, exercise_type):
        self.calibrator = PoseCalibrator(model_path='yolov8n-pose.pt')
        self.metrics = PerformanceMetrics()
        self.exercise_type = exercise_type
        self.counter = 0  
        self.stage = None 
        
        # Angle thresholds lagiye
        self.thresholds = {
            'pushup': {'down': 90, 'up': 160, 'form_hip_min': 150},
            'squat': {'down': 100, 'up': 160, 'deep': 80},
            'situp': {'up': 70, 'down': 20, 'good_crunch': 50}
        }
        self.metrics.exercise = exercise_type

    def process_pushup(self, angles, keypoints):
        """Process pushup exercise using elbow angle detection"""
        elbow = angles['left_elbow'] if keypoints[7][2] > keypoints[8][2] else angles['right_elbow']
        hip = angles['left_hip'] if keypoints[7][2] > keypoints[8][2] else angles['right_hip']
        if elbow is None or hip is None: return

        self.metrics.update_angle_data(angles.get('left_elbow'), angles.get('right_elbow'), 
                                       angles.get('left_hip'), angles.get('right_hip'),
                                       None, None, None, None, None, None, None, None)

        if elbow > self.thresholds['pushup']['up']: self.stage = "UP"
        if elbow < self.thresholds['pushup']['down'] and self.stage == 'UP':
            self.stage = "DOWN"
            self.counter += 1
            self.metrics.record_rep(self.thresholds['pushup']['up'], elbow, 1.0, 
                                   hip >= self.thresholds['pushup']['form_hip_min'])

    def process_squat(self, angles, keypoints):
        """Process squat exercise using knee angle and timing"""
        knee = angles['left_knee'] if keypoints[13][2] > keypoints[14][2] else angles['right_knee']
        if knee is None: return
        
        # Track time
        current_time = time.time()
        torso_angle = angles.get('torso_angle')
        shin_angle = angles.get('shin_angle_left') if keypoints[13][2] > keypoints[14][2] else angles.get('shin_angle_right')
        self.metrics.update_squat_data(keypoints, angles, torso_angle, shin_angle, current_time)
        
        # Standing position (knee extended)
        if knee > self.thresholds['squat']['up']:
            if self.stage == "DOWN" and self.metrics.rep_bottom_time:
                # Calculate concentric (up) phase time
                self.metrics.concentric_times.append(current_time - self.metrics.rep_bottom_time)
                self.metrics.rep_bottom_time = None
            self.stage = "UP"
            self.metrics.current_phase = 'standing'
            if self.metrics.rep_start_time is None: self.metrics.rep_start_time = current_time
        # Bottom position (knee bent) - count rep
        elif knee < self.thresholds['squat']['down'] and self.stage == 'UP':
            self.stage = "DOWN"
            self.counter += 1
            if self.metrics.rep_start_time:
                # Calculate eccentric (down) phase time
                self.metrics.eccentric_times.append(current_time - self.metrics.rep_start_time)
            self.metrics.rep_bottom_time = current_time
            self.metrics.rep_start_time = None
            self.metrics.squat_depths.append(knee)
            # Check form quality (depth and torso angle)
            is_good = knee < self.thresholds['squat']['deep'] and (not torso_angle or torso_angle <= 45)
            if is_good: self.metrics.good_reps += 1
            else: self.metrics.bad_reps += 1
            self.metrics.record_rep(self.thresholds['squat']['up'], knee, 1.0, is_good)

    def process_situp(self, angles, keypoints):
        """Process situp exercise using torso inclination and state machine"""
        current_time = time.time()
        torso_inclination = angles.get('torso_inclination_horizontal')
        hip_flexion = angles.get('hip_flexion_angle')
        if torso_inclination is None: return
        
        # Update metrics and detect form violations
        self.metrics.update_situp_data(keypoints, angles, torso_inclination, hip_flexion, current_time)
        foot_lifted = self.metrics._detect_foot_lift(keypoints)
        
        # Rest position (lying down)
        if torso_inclination <= self.thresholds['situp']['down']:
            if self.metrics.situp_state == 'descending' and self.metrics.situp_peak_time:
                # Calculate eccentric (down) phase time
                self.metrics.situp_eccentric_times.append(current_time - self.metrics.situp_peak_time)
                self.metrics.situp_peak_time = None
                self.metrics.situp_momentum_scores.append(self.metrics._calculate_momentum_score())
                self.metrics.shoulder_positions.clear()
            self.metrics.situp_state = 'rest'
            self.stage = "DOWN"
            if self.metrics.situp_rep_start_time is None: self.metrics.situp_rep_start_time = current_time
        # Peak position (sitting up) - count rep
        elif torso_inclination >= self.thresholds['situp']['up'] or (hip_flexion and hip_flexion <= self.thresholds['situp']['good_crunch']):
            if self.metrics.situp_state in ['rest', 'ascending']:
                self.counter += 1
                self.metrics.situp_state = 'peak'
                self.stage = "UP"
                if self.metrics.situp_rep_start_time:
                    # Calculate concentric (up) phase time
                    self.metrics.situp_concentric_times.append(current_time - self.metrics.situp_rep_start_time)
                    self.metrics.situp_rep_start_time = None
                self.metrics.situp_peak_time = current_time
                # Record rep data
                self.metrics.situp_torso_inclinations.append(self.metrics.max_torso_inclination)
                self.metrics.situp_hip_flexions.append(self.metrics.min_hip_flexion if hip_flexion else 180)
                self.metrics.situp_foot_lifts.append(1 if foot_lifted else 0)
                # Check form quality
                good_rom = torso_inclination >= self.thresholds['situp']['up']
                good_crunch = hip_flexion and hip_flexion <= self.thresholds['situp']['good_crunch']
                is_good = good_rom and not foot_lifted
                if is_good: 
                    self.metrics.good_reps += 1
                    self.metrics.situp_valid_reps += 1
                else: 
                    self.metrics.bad_reps += 1
                    if not good_rom: self.metrics.situp_short_rom_count += 1
                self.metrics.record_rep(torso_inclination, 0, 1.0, is_good)
            # Transition to descending phase
            if torso_inclination < self.thresholds['situp']['up'] - 10:
                self.metrics.situp_state = 'descending'
        # Intermediate positions - update state
        else:
            if self.metrics.situp_state == 'rest': self.metrics.situp_state = 'ascending'
            elif self.metrics.situp_state == 'peak': self.metrics.situp_state = 'descending'

    def process_frame(self, frame):
        """Process a single video frame for pose detection and exercise analysis"""
        processed_frame, keypoints, angles = self.calibrator.process_frame(frame, show_angles_panel=False)
        if keypoints is not None:
            # Route to appropriate exercise processor
            if self.exercise_type == 'pushup': self.process_pushup(angles, keypoints)
            elif self.exercise_type == 'squat': self.process_squat(angles, keypoints)
            elif self.exercise_type == 'situp': self.process_situp(angles, keypoints)
        return processed_frame

    def get_final_metrics(self):
        """Calculate and return final exercise metrics"""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            if self.exercise_type == 'pushup': 
                result = self.metrics.pushup_metrics()
            elif self.exercise_type == 'squat': 
                result = self.metrics.squat_metrics()
            elif self.exercise_type == 'situp': 
                result = self.metrics.situp_metrics()
            else: 
                result = {}
        finally:
            sys.stdout = old_stdout
        return result
# api ka nanga nach(eto sundor code ai o likte parbena laura)
@app.post("/api/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), exercise_type: str = "pushup"):
    try:
        job_id = str(uuid.uuid4())
        file_path = f"uploads/{job_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        background_tasks.add_task(process_video_background, file_path, exercise_type, job_id)
        return JSONResponse({'success': True, 'job_id': job_id})
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

def process_video_background(file_path: str, exercise_type: str, job_id: str):
    try:
        session = ExerciseSession(exercise_type)
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_results[job_id] = {'status': 'processing', 'progress': 0}
        # 3 te frame chara progress check koreche for better performance
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 3 == 0: session.process_frame(frame)
            frame_count += 1
            video_results[job_id]['progress'] = int((frame_count / total_frames) * 100)
        
        cap.release()
        video_results[job_id] = {
            'status': 'completed',
            'progress': 100,
            'metrics': session.get_final_metrics(),
            'reps': session.counter,
            'timestamp': datetime.now().isoformat()
        }
        if os.path.exists(file_path): os.remove(file_path)
    except Exception as e:
        video_results[job_id] = {'status': 'failed', 'error': str(e)}

@app.get("/api/video-status/{job_id}")
async def get_video_status(job_id: str):
    if job_id not in video_results:
        return JSONResponse({'error': 'Job not found'}, status_code=404)
    return JSONResponse(video_results[job_id])
# fully ai (bal kaj korbe eta)
@app.post("/api/ai-feedback")
async def get_ai_feedback(metrics: dict):
    try:
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI
        load_dotenv()
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_retries=2)
        messages = [
            ("system", "You are a fitness coach. Provide brief, actionable feedback."),
            ("human", f"Analyze these metrics and give 3 key tips:\n\n{json.dumps(metrics, indent=2)}"),
        ]
        ai_msg = llm.invoke(messages)
        return JSONResponse({'success': True, 'feedback': ai_msg.content})
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get("/api/health")
async def health_check():
    return JSONResponse({'status': 'healthy', 'version': '1.0.0', 'timestamp': datetime.now().isoformat()})

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)