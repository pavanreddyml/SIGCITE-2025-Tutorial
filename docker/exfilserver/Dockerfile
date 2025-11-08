FROM python:3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnspr4 libnss3 libdbus-1-3 \
    libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libxkbcommon0 libasound2t64 \
    wget ca-certificates fonts-liberation tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN pip install flask flask-cors pillow playwright html2text requests
RUN python -m playwright install --with-deps chromium

COPY . .

EXPOSE 2500

CMD ["python", "bsidesnova/bsidesnova/server/exfil_server.py"]
    