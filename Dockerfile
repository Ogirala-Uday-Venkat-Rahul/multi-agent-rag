# Hugging Face Spaces (Docker SDK) compatible image.
# HF runs the container as uid 1000 and routes to the port in app_port (7860).
FROM python:3.11-slim

# Non-root user, as HF Spaces expects
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/home/user/app/.cache

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
