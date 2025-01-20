from crewai import Agent, Task, Crew , LLM
from langchain_groq import ChatGroq
from crewai_tools import YoutubeVideoSearchTool
from pytube import exceptions
from langchain_community.document_loaders import YoutubeLoader
import os
import asyncio
import edge_tts
from diffusers import StableDiffusionPipeline
import torch
import time
from moviepy import ImageClip , AudioFileClip , CompositeVideoClip , concatenate_videoclips , VideoFileClip , CompositeAudioClip , VideoClip , TextClip
import whisper_timestamped as whisper
import ffmpeg

os.environ['GROQ_API_KEY'] = "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"


for m in range(3):
    story_writer = Agent(
        role="Video story writer",
        goal="Write a 120 words horror story for youtube shorts ",
        backstory="""You are a highly skilled horror story/script writer that writes stories and scripts for youtube shorts
        which are generally upto 120 words. Your stories are crisp, concise and interesting.""",
        verbose=True,
        allow_delegation=False,
        llm= "groq/llama3-70b-8192"
    )

    # Create the Analysis Report Writer Agent
    Prompt_generator = Agent(
        role="Prompt generator for AI art generators",
        goal="Generate a prompt of maximum 65 tokens from the story given to you, which can be used to generate images from ai art generators",
        backstory="""You are an experienced prompt generator that can generate prompts of maximum 65 tokens from the stories provided to you so that they could be used 
        to generate images from ai art generators.""",
        verbose=True,
        allow_delegation=False,
        llm = "groq/llama3-70b-8192"
    )

    # Create the Transcript Analysis Task
    story_writer_task = Task(
        description=""" Write a horror story of about 120 words that could be used to generate youtube shorts. Make sure-> 
        1. The theme should be horror.
        2. Stories should be upto 140 words
        3. Should be fun for the audience
        4. Should conatin only the title and story 
        """,
        agent=story_writer,
        expected_output="""A structured horror story for youtube shorts that are:
        - Fun
        - Exciting
        - 120 words
        - Spooky"""
    )

    # Create the Analysis Report Writing Task
    prompt_writing_task = Task(
        description="""Using the story {story}, create a maximum 65 token prompt for ai image generator, make sure:
        1. Clearly structured
        2. 65 tokens max
        3. Written for an ai generator""",
        agent=Prompt_generator,
        expected_output="""A well-formatted prompt that:
        - Breaks down the story's content
        - Is of 65 tokekns """
    )

    # Create and Configure the Crew
    crew = Crew(
        agents=[story_writer],
        tasks=[story_writer_task],
        verbose=True,share_crew=True
    )

    crew2 = Crew(
        agents=[Prompt_generator],
        tasks=[prompt_writing_task],
        verbose=True,share_crew=True
    )

    result = crew.kickoff()

    a = crew2.kickoff(inputs={'story' : result})

    TEXT = str(result)

    VOICE = "en-AU-WilliamNeural"

    OUTPUT_FILE = f"horror{m}.mp3"

    async def amain() -> None:
        communicate = edge_tts.Communicate(TEXT, VOICE)
        await communicate.save(OUTPUT_FILE)

    asyncio.run(amain())

    audio = AudioFileClip(f"horror{m}.mp3") # Replace with your audio file path

    # Get the duration
    audio_length = audio.duration

    length = int(audio_length)

    background_music = AudioFileClip("bgm.mp3")

    l = length + 6

    background_music = background_music.subclipped(0, l)  
    background_music = background_music.with_volume_scaled(0.2)


    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")


    for i in range(3):
        prompt = str(a)
        image = pipe(prompt).images[0]
        image.save(f"./result{i}.jpg")
        time.sleep(5)


    def resize_func(t):
        return 1 + 0.02*t 
            # Zoom-in.


    d = length/3
    duration = d + 2

    screensize = (1080,1920)

    clips = []

    for i in range(3):
        path = f"C:/Users/SARTHAK/Desktop/shorts/result{i}.jpg"

        clip_img = (
        ImageClip(path)
        .resized(screensize)
        .resized(resize_func)
        .with_position(('center', 'center'))
        .with_duration(duration)
        .with_fps(24)
        )

        clip = CompositeVideoClip([clip_img], size=screensize)
        clip.write_videofile(f'test{i}.mp4')
        clips.append(clip)


    final_video = concatenate_videoclips(clips, method="compose")

    # Replace with your audio file

    # Set the audio to the video





    final_video = final_video.with_audio(audio)



    # Export the final video
    final_video.write_videofile('tes.mp4')

    final_video = VideoFileClip('tes.mp4')

    if final_video.audio:
        main_audio = final_video.audio  # Extract original video audio
        final_audio = CompositeAudioClip([main_audio, background_music])  # Combine both audios
    else:
        final_audio = background_music  # If no main audio exists, use only the background music

    # Set the final audio to the video
    final_video = final_video.with_audio(final_audio)

    # Export the final video
    final_video.write_videofile("output_with_background_music.mp4")


    filename = r"C:\Users\SARTHAK\Desktop\shorts\output_with_background_music.mp4"

    screen_width = 1080
    screen_height = 1920

    def get_transcribed_text(filename):
        audio = whisper.load_audio(filename)
        model = whisper.load_model("small", device="cpu")
        results = whisper.transcribe(model, audio, language="en")

        return results["segments"]

    font_path = r"C:\Windows\Fonts\Arial.ttf"

    def get_text_clips(text, fontsize):
        text_clips = []
        for segment in text:
            for word in segment["words"]:
                text_clip = TextClip(
                    text=word["text"],  # Changed 'word["text"]' to txt=word["text"]
                    font_size =fontsize,
                    size = (800,800) ,
                    method='caption',
                    stroke_width=5, 
                    stroke_color="black", 
                    font = font_path,
                    color="white"
                ).with_start(word["start"]).with_end(word["end"]).with_position("center")
                text_clips.append(text_clip)
        return text_clips

    # Loading the video as a VideoFileClip
    original_clip = VideoFileClip(filename)

    # Load the audio in the video to transcribe it and get transcribed text
    transcribed_text = get_transcribed_text(filename)
    # Generate text elements for video using transcribed text
    text_clip_list = get_text_clips(text=transcribed_text, fontsize=150)
    # Create a CompositeVideoClip that we write to a file
    final_clip = CompositeVideoClip([original_clip] + text_clip_list)

    final_clip.write_videofile(f"final1{m}.mp4", codec="libx264")

    generator = Agent(
        role="Title and tag generator for youtube",
        goal="Generate a title and tags from the story given to you, which can be used to upload on youtube",
        backstory="""You are an experienced title generator and tags generator for the purpose of youtube uploading.""",
        verbose=True,
        allow_delegation=False,
        llm = "groq/llama3-70b-8192"
    )


    writing_task = Task(
        description="""Using the story {story}, create a title and tags for the story so that it could be uploaded to youtbe, make sure:
        1. Title is not too big, just 10 words max 
        2. Generate minimum 30 tags
        3. Sensible tags and title
        4. the tags are written with a '#' and not a list,""",
        agent=generator,
        expected_output="""A well-formatted title and tags that:
        - Explain the story
        - Helps the video to reach its maximum potential
        - tags are written with '#' and not a list manner """
    )

    # Create and Configure the Crew
    crew3 = Crew(
        agents=[generator],
        tasks=[writing_task],
        verbose=True,share_crew=True
    )

    b = crew3.kickoff(inputs={'story' : result})

    content = str(b)

    file_name = f"generated_content{m}.txt"

    # Create and write content to the file
    with open(file_name, 'w') as file:
        file.write(content)

    print(f"File '{file_name}' has been created and content added.")








