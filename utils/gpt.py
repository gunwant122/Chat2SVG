import os
import yaml
import base64
import mimetypes
import requests
from dotenv import load_dotenv
from utils.util import read
from google import genai
from google.genai import types

# Create a .env file in the project root directory and add your Anthropic API key as ANTHROPIC_API_KEY=<your_key>
load_dotenv(dotenv_path=os.path.join("..", ".env"))
api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


class Session:
    def __init__(self, model, prompts_file) -> None:
        self.model = model
        self.past_tasks: list[str] = []
        self.past_messages = []
        self.past_responses: list[str] = []

        # Load the predefined prompts for the LLM
        with open(f"{prompts_file}.yaml") as file:
            self.predefined_prompts: dict[str, str] = yaml.safe_load(file)

    def send(self, task: str, prompt_info: dict[str, str] | None = None, images: list[str] = [], file_path=None) -> str:
        print(f"$ --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        response = self._send(prompt, images, file_path)
        print(f"$ --- Response:\n{response}\n")
        return response

    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None) -> tuple[str, str]:
        # Get the predefined prompt for the task
        user_prompt = ""
        system_prompt = self.predefined_prompts["system"] + self.predefined_prompts[task]

        # Check for task-specific required information
        valid = True
        match task:
            case "expand_text_prompt":
                valid = "text_prompt" in prompt_info
                user_prompt = "Here is the text prompt: "
                system_prompt= self.predefined_prompts["system"] + self.predefined_prompts[task]
            case _:
                user_prompt = self.predefined_prompts[task]
                system_prompt = self.predefined_prompts["system"]

        if not valid:
            raise ValueError(f"Extra information is required for the task: {task}")

        # Replace the placeholders in the prompt with the information
        print(f"$ --- Prompt Info: {prompt_info}")
        if prompt_info is not None:
            for key in prompt_info:
                user_prompt = user_prompt + f" {prompt_info[key]}\n"
        print(f"$ --- User Prompt: {user_prompt}")
        # print(f"$ --- System Prompt: {system_prompt}")
        return system_prompt, user_prompt

    def _send(self, prompt: tuple[str, str], images: list[str] = [], file_path=None) -> str:
        if self.model.startswith("gemini"):
            return self._send_gemini(prompt, images, file_path)
            
        system_prompt, user_prompt = prompt
        payload = self._create_payload(user_prompt, images=images)
        if not os.path.exists(file_path):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            print("Waiting for LLM to be generated")
            response = requests.post("https://api.gptsapi.net/v1/chat/completions", headers=headers, json=payload,
                                     timeout=(5, 120))  # WildCard
            print(f"LLM response: {response.text}")
            try:
                response = response.json()['choices'][0]['message']['content']
            except:
                print(f"$ --- Error Response: {response.json()}\n")
                # Ensure we're returning a string, not a Response object
                response = str(response.text)
        else:
            response = read(file_path)
        self.past_messages.append({"role": "assistant", "content": response})
        self.past_responses.append(response)
        return response

    def _send_gemini(self, prompt: tuple[str, str], images: list[str] = [], file_path=None) -> str:
        if file_path and os.path.exists(file_path):
            return read(file_path)

        client = genai.Client(api_key=gemini_api_key)
        system_prompt, user_prompt = prompt
        print("User Prompt:", user_prompt)
        print("System Prompt:", system_prompt)
        
        # Add formatting instruction to the prompt
        formatted_prompt =user_prompt 
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=formatted_prompt),
                ],
            ),
        ]

        print("Contents:", contents)
        # Add images if any
        for image in images:
            with open(image, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                contents[0].parts.append(
                    types.Part.from_bytes(
                        mime_type="image/png",
                        data=image_data
                    )
                )

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",  # Changed from application/json to text/plain
            system_instruction=[
                types.Part.from_text(text=system_prompt),
            ],
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )
            
            response_text = response.text
            print("Gemini response:", response_text)
            
            # Extract SVG code from between ```svg and ``` tags
            import re
            svg_match = re.search(r'```svg\n(.*?)\n```', response_text, re.DOTALL)
            if svg_match:
                response_text = svg_match.group(1).strip()
            
            self.past_messages.append({"role": "user", "content": response_text})
            self.past_responses.append(response_text)
            return response_text
        except Exception as e:
            print(f"$ --- Error Response from Gemini: {str(e)}\n")
            error_response = str(e)
            self.past_messages.append({"role": "user", "content": error_response})
            self.past_responses.append(error_response)
            return error_response

    def _create_payload(self, prompt: str, images: list[str] = []):
        """Creates the payload for the API request."""
        messages = {
            "role": "user",
            "content": [],
        }

        for image in images:
            base64_image = encode_image(image)
            image_content = {
                ## ChatGPT
                # "type": "image_url",
                # "image_url": {
                #     "url": base64_image,
                #     "detail": "auto",
                # }
                # Claude
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': base64_image
                }
            }
            messages["content"].append(image_content)

        messages["content"].append({
            "type": "text",
            "text": prompt,
        })

        self.past_messages.append(messages)

        return {
            "model": self.model,
            "system": self.predefined_prompts["system"],
            "messages": self.past_messages,
        }


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # return f"data:{mime_type};base64,{encoded_string}"  # ChatGPT
        return encoded_string  # Claude
