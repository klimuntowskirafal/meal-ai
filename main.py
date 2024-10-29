from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import aiofiles
import os
from PIL import Image
import io
import glob
import anthropic
import base64
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()  # Load environment variables

app = FastAPI()

@app.get("/")
async def root():
    await asyncio.sleep(2)
    return {"message": "Hello, wordl  yolo!"}

@app.post("/estimate-calories")
async def estimate_calories(file: UploadFile = File(...)):
    # Ensure the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    # Read and save the file
    try:
        async with aiofiles.open(f"temp_{file.filename}", "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Process the image (placeholder for actual image processing)
    try:
        img = Image.open(f"temp_{file.filename}")
        # Here you would use your image analysis model to identify food items
        
        # Placeholder for calorie estimation
        estimated_calories = 500  # This should be replaced with actual estimation
        
        # Clean up the temporary file
        os.remove(f"temp_{file.filename}")
        
        return JSONResponse(content={
            "filename": file.filename,
            "estimated_calories": estimated_calories,
            "confidence": 0.8  # This should be provided by your model
        })
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

async def test_calorie_estimation():
    # Look for image files with various extensions
    image_patterns = ["meal1.*", "food1.*"]  # Add more patterns if needed
    image_path = None

    for pattern in image_patterns:
        matching_files = glob.glob(pattern)
        if matching_files:
            image_path = matching_files[0]
            break

    if not image_path:
        print("Error: No suitable image file found in the root directory.")
        return

    print(f"Using image: {image_path}")

    # Open and prepare the image
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB (in case it's not)
            img = img.convert('RGB')
            # Create a byte stream of the image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return

    # Test different LLMs
    llms_to_test = [
        test_openai_gpt,
        # test_google_palm,
        # test_anthropic_claude,
        # Add more LLM test functions as needed
    ]

    for llm_test in llms_to_test:
        try:
            result = await llm_test(img_byte_arr)
            print(f"Results from {llm_test.__name__}:")
            print(result)
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"Error with {llm_test.__name__}: {str(e)}")

async def test_openai_gpt(image_bytes):
    # Convert image bytes to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Initialize OpenAI client through LangChain with updated model name
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Updated model name
        max_tokens=300,
        temperature=0.7
    )

    # Prepare the message with proper image formatting
    message = HumanMessage(
        role="user",
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"  # Added detail level
                }
            },
            {
                "type": "text",
                "text": "Please analyze this meal image and provide: \
                1. A list of identified food items \
                2. Estimated portion sizes for each item \
                3. Estimated calories for each item \
                4. Total estimated calories for the entire meal \
                5. Confidence level in your estimation (low/medium/high) \
                Please format the response as JSON."
            }
        ]
    )

    try:
        # Make the API call
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        detailed_error = f"Error calling OpenAI API: {str(e)}"
        print(detailed_error)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image with OpenAI: {str(e)}"
        )

async def test_google_palm(image_bytes):
    # Implement Google PaLM
    # You'll need to set up the Google Cloud credentials
    # This is a placeholder implementation
    return "Google PaLM calorie estimation placeholder"

async def test_anthropic_claude(image_bytes):
    print('test_anthropic_claude execution')
    
    # Convert image bytes to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Initialize Anthropic client through LangChain
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", max_tokens=1000)
    
    # Prepare the message
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": base64_image
                }
            },
            {
                "type": "text",
                "text": "Please analyze this meal image and provide: \
                1. A list of identified food items \
                2. Estimated portion sizes for each item \
                3. Estimated calories for each item \
                4. Total estimated calories for the entire meal \
                5. Confidence level in your estimation (low/medium/high) \
                Please format the response as JSON."
            }
        ]
    )

    try:
        # Make the API call
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image with Claude: {str(e)}")

# Add this to your FastAPI app
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(test_calorie_estimation())



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
