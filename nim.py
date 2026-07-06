from openai import OpenAI
import sys
import time
start_time = time.perf_counter()


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)


#message = " ".join(sys.argv[1:]).strip()
#if not message:
#  message = input("Message: ").strip()
#if not message:
 # raise SystemExit("No message provided.")


completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-pro",
  messages=[{"role":"user","content":"You are simulating one day in the life of a character in a generative-agents "
        "simulation. Produce a COARSE day plan: 5 to 8 broad blocks of activity covering "
        "the full 24 hours (00:00 to 24:00), with no gaps and no overlaps. "
        "Stay true to the persona's traits and background.\n\n"
        "For EACH block, tag it as 'atomic' or 'flexible':\n"
        "- atomic: a single continuous activity with no meaningful internal sub-steps "
        "worth planning separately. Examples: sleeping, attending a class/lecture, "
        "commuting/travel, sitting an exam, watching a movie, a long uninterrupted "
        "study/deep-work session.\n"
        "- flexible: an activity that naturally contains distinct sub-activities a person "
        "would actually think of as separate steps. Examples: 'morning routine' (wake up, "
        "shower, get dressed), 'gym session' (warm-up, lifting, cooldown), 'dinner with "
        "friends' (walk over, eat, chat).\n"
        "When in doubt, prefer 'atomic' -- do not manufacture sub-steps for something a "
        "person would just describe as one thing."}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"thinking":False}},
  stream=True
)

for chunk in completion:
  if not getattr(chunk, "choices", None):
    continue
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


    end_time = time.perf_counter()

    execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
