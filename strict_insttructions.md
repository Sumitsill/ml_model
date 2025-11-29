
kire bara .md file dekhei chole eli .

how to start?
 
====> 
1. you should have python 3.11 or you should die.
2. run in the terminal " pip install -r requiments.txt "
3. double click on the start.bat
   or 
   run "python start_server.py"

### api endpoints

 2. Upload Video****

POST /api/upload-video?exercise_type=pushup
Body: multipart/form-data with video file
Returns: {"job_id": "uuid"}
 
 ## url: http://localhost:8000/api/upload-video?exercise_type=pushup' \

3. Check Status

GET /api/video-status/{job_id}
Returns: {"status": "processing|completed", "progress": 0-100, "metrics": {...}}
 ## url: http://localhost:8000/api/video-status/{job_ID}

5. AI Feedback

POST /api/v1/ai-feedback
Body: metrics JSON


6. Health Check

GET /api/v1/health


ar bhalo lagche na 

see the fast api docs to nunderstand evrything ,love you . welm=come to my world