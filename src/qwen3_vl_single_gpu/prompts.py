from __future__ import annotations


ROAMII_PROMPT_V4_SYSTEM = "You are a scenic spot image matching expert."

ROAMII_PROMPT_V4_USER = (
    "Do these two images show the same scenic subject?\n\n"
    "First identify the subject type, then apply the matching rule:\n\n"
    "(A) Architecture or large scene (building, wall, garden, plaza):\n"
    "- Match by the structure's form, layout, and distinctive details.\n"
    "- Different viewpoints, distances, or lighting of the same structure count as a match.\n"
    "- Do NOT be misled by similar interior style or atmosphere - different rooms can look alike.\n"
    "- If visible text or signage differs between images, they are definitely different subjects.\n\n"
    "(B) Exhibit or small object (statue, sculpture, stone, artifact, inscription):\n"
    "- Match by the object's specific form: shape, posture, gesture, proportions, and surface details.\n"
    "- Two different objects in a similar display setting are NOT a match.\n"
    "- Any confirmed difference in form - different base, posture, hand position, headwear, or\n"
    "  ornamentation - means 'No', even if they look very similar overall.\n\n"
    "Answer only 'Yes' or 'No'."
)
