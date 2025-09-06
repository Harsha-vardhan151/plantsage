FROM node:20-alpine
WORKDIR /app
COPY package.json /app/
RUN npm install --omit=dev
COPY . /app
ENV PORT=8080
CMD ["npm","start"]