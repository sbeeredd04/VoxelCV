import pathlib
import cv2
from google import genai
import PIL.Image
import time

# Initialize Gemini
GOOGLE_API_KEY = "AIzaSyCOw3F-FxagrfJE-hBjBeIjGIsKYINHC1k"
genai.configure(api_key=GOOGLE_API_KEY)

def analyze_fall_with_gemini(image_path):
    """Analyze image for fall detection using Gemini"""
    try:
        # Load image
        image = PIL.Image.open(image_path)
        
        # Create Gemini client
        client = genai.GenerativeModel('gemini-pro-vision')
        
        # Craft specific prompt for fall detection
        prompt = """
        Analyze this image for potential fall detection. Consider:
        1. Is the person in a normal standing/sitting position?
        2. Is the person in an unusual position that might indicate a fall?
        3. Is the person lying on the ground unexpectedly?
        4. Are there any signs of distress or unusual body positioning?
        
        Respond with one of these categories:
        - NORMAL: Person is in a normal standing, sitting, or intentionally lying position
        - FALLING: Person appears to be in the process of falling
        - FALLEN: Person appears to have fallen and is in a potentially dangerous position
        
        Also provide a brief explanation for your classification.
        """

        # Generate response
        response = client.generate_content([prompt, image])
        
        # Process and format response
        result = response.text
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'timestamp': timestamp,
            'analysis': result,
            'status': 'success'
        }

    except Exception as e:
        return {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'analysis': f"Error analyzing image: {str(e)}",
            'status': 'error'
        }

def analyze_and_display(image_path):
    """Analyze a single image and display results"""
    # Get analysis from Gemini
    result = analyze_fall_with_gemini(image_path)
    
    # Load and display image with results
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Create a copy for annotation
    annotated_image = image.copy()
    
    if result['status'] == 'success':
        # Display result on image
        lines = result['analysis'].split('\n')
        y_position = 30
        
        for line in lines:
            if line.strip():  # Only process non-empty lines
                # Determine color based on classification
                color = (0, 255, 0)  # Default green
                if 'FALLING' in line:
                    color = (0, 255, 255)  # Yellow
                elif 'FALLEN' in line:
                    color = (0, 0, 255)  # Red
                
                # Add text to image
                cv2.putText(annotated_image, line.strip(), (10, y_position),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_position += 25
        
        # Add timestamp
        cv2.putText(annotated_image, f"Time: {result['timestamp']}", 
                   (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display images side by side
        combined_image = cv2.hconcat([image, annotated_image])
        cv2.imshow('Original | Analysis', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print analysis to console
        print("\nFall Detection Analysis:")
        print(f"Timestamp: {result['timestamp']}")
        print("Analysis:")
        print(result['analysis'])
        
    else:
        print("Error in analysis:", result['analysis'])

if __name__ == "__main__":
    import sys
        
    image_path = 
    analyze_and_display(image_path)
