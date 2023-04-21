# FROM node:14.20.0-alpine3.15 as builder
# RUN apk add --no-cache chromium
# COPY . /app
# WORKDIR /app
# RUN npm install
# RUN npm run build

# FROM nginx:1.17.10-alpine
# EXPOSE 80
# COPY --from=builder /app/dist/apache-air-app /usr/share/nginx/html

# Create image based on the official Node 10 image from dockerhub
FROM node:14.20.0-alpine3.15

# Create a directory where our app will be placed
RUN mkdir -p /app

# Change directory so that our commands run inside this new directory
WORKDIR /app

# Copy dependency definitions
COPY package*.json /app/

# Install dependecies
RUN npm install

# Get all the code needed to run the app
COPY . /app/

# Expose the port the app runs in
EXPOSE 4200

# Serve the app
CMD ["npm", "start"]