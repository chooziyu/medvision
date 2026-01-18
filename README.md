# MedVision

## Inspiration

Medical video feeds such as ultrasound, echocardiography, and laparoscopy are inherently dynamic due to camera motion and natural anatomical movement. While many existing collaboration and annotation tools allow users to mark up these videos, the annotations are typically static and quickly become misaligned as the video moves, limiting their usefulness in real clinical and educational settings.

Inspired by HoloRay’s mission to improve healthcare collaboration and learning, **MedVision** explores how **motion-tracked annotations** can make medical video interpretation clearer, more intuitive, and more reliable.

## What It Does

MedVision allows users to place annotations directly onto a medical video feed and keeps those annotations anchored to the same underlying anatomy as the video moves.

Instead of annotations “floating” or drifting when the camera or anatomy shifts, each annotation follows the motion of the visual structure it was placed on. This demonstrates how motion-tracked annotation can significantly improve clarity during live or recorded medical procedures.

## How We Built It

MedVision is built using a modular pipeline that adapts the tracking approach to the imaging modality, allowing us to balance robustness, performance, and real-time constraints.

### Echocardiography (YOLO Segmentation)

For echocardiography, we use **YOLO-based segmentation** to identify and track anatomical structures.

YOLO (You Only Look Once) segmentation models work by processing the entire image in a single forward pass of a convolutional neural network. Instead of scanning the image region by region, YOLO predicts:

* Object classes
* Bounding boxes
* Pixel-level segmentation masks

This makes YOLO particularly well-suited for real-time medical video, where both speed and spatial accuracy are important. In echocardiography, segmentation masks allow annotations to be anchored to deformable cardiac structures rather than rigid points, enabling them to follow natural anatomical motion throughout the cardiac cycle.

The segmentation output is used to associate user annotations with specific regions of interest. As the segmented region moves or deforms between frames, the annotation is updated accordingly, maintaining alignment with the underlying anatomy.

### Microscopy (Lucas–Kanade Optical Flow)

For microscopy video, we rely on the **Lucas–Kanade optical flow algorithm** to track user-selected points.

Lucas–Kanade estimates motion by assuming that pixel intensities remain consistent between consecutive frames and that motion within a small neighborhood is approximately constant. By solving a local least-squares problem, the algorithm computes how selected feature points move frame to frame.

This approach works well for microscopy data, where motion is typically small, smooth, and locally coherent. Once a user places an annotation, the underlying feature points are tracked over time, allowing the annotation to remain anchored to the same microscopic structure.

### Laparoscopy (YOLO Bounding Boxes)

For laparoscopy, we use **YOLO with pretrained bounding box detection**.

The YOLO detector identifies surgical tools or anatomical regions using bounding boxes rather than pixel-level segmentation. User annotations are associated with these detected boxes, and as the bounding boxes move across frames, the annotations follow accordingly.

This approach is computationally efficient and robust to larger camera motions commonly seen in laparoscopic footage, making it suitable for real-time tracking scenarios.

### Frontend / UI

* HTML, CSS, and JavaScript
* Video playback with annotation overlays
* Visualization of tracking states and confidence

### Real-Time Constraints and Latency

The video feed captures input frames in real time; however, there are inherent **latencies** introduced by:

* Model inference time (YOLO detection and segmentation)
* Python-to-frontend communication
* Rendering and synchronization with the display frame rate

These latencies can cause slight delays between the video motion and annotation updates. Managing this trade-off between accuracy, robustness, and real-time responsiveness was a key design consideration throughout the project.

## Challenges We Ran Into

* Designing tracking that works well with **user-placed annotations**, which is significantly harder than tracking predefined objects
* Maintaining precision and stability in the presence of:

  * Jitter and drift
  * Occlusion
  * Annotated regions temporarily leaving the frame
* Integrating Python-based tracking logic with frontend visualization
* Ensuring smooth, real-time updates that stayed synchronized with the video frame rate

## Accomplishments

We’re particularly proud of:

* Tackling a technically challenging computer vision problem as a team of beginners
* Successfully collaborating across frontend and backend roles
* Designing a UI that clearly demonstrates tracking accuracy, stability, and robustness
* Completing our first hackathon project while learning new tools under significant time pressure

## What We Learned

Through building MedVision, we gained hands-on experience with:

* Motion tracking and core computer vision concepts in Python
* OpenCV feature tracking and optical flow techniques
* Rapid prototyping and problem-solving in a hackathon environment
* Effective team collaboration under time constraints

## What’s Next

Future improvements we’d like to explore include:

* Supporting **multiple annotations** simultaneously
* Improving tracking robustness using more advanced models
* Enabling **collaborative, multi-user annotation** for shared learning and remote clinical discussion

---

*MedVision was built as a hackathon project to explore how motion-tracked annotations can improve clarity and collaboration in medical video interpretation.*
