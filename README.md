\# SignalScope



\## AI-Generated Image Detection System



SignalScope is an AI-powered image authenticity detection system designed to estimate whether an uploaded image is likely AI-generated or likely real.



The system uses a deep-learning image classifier together with Grad-CAM visual explanations and a web-based interface.



> SignalScope provides a likelihood-based assessment and should not be treated as definitive proof of image authenticity.



\---



\# 1. Problem Statement



AI image generators are becoming increasingly capable of producing realistic images.



This creates a need for systems that can analyze images and estimate whether they were generated synthetically or captured from the real world.



SignalScope addresses this problem using a computer-vision classifier trained on real and AI-generated images from multiple generator families.



\---



\# 2. Main Features



\- AI-generated vs real image classification

\- AI probability

\- Real probability

\- Confidence score

\- Likely AI-generated / Likely real verdict

\- Grad-CAM visual explanation

\- Image upload through web interface

\- Drag-and-drop image upload

\- Prediction history

\- Clear history functionality

\- Robustness evaluation

\- Multiple-generator evaluation

\- External distribution-shift evaluation

\- FastAPI backend

\- React frontend



\---



\# 3. System Architecture



```text

&#x20;                   Uploaded Image

&#x20;                         |

&#x20;                         v

&#x20;               Image Preprocessing

&#x20;                         |

&#x20;                         v

&#x20;                   ResNet18 CNN

&#x20;                         |

&#x20;                         v

&#x20;                Probability Output

&#x20;                         |

&#x20;            +------------+------------+

&#x20;            |                         |

&#x20;            v                         v

&#x20;      AI / Real Verdict          Grad-CAM

&#x20;            |                         |

&#x20;            +------------+------------+

&#x20;                         |

&#x20;                         v

&#x20;                   Web Interface

