# Stage 1: Build the React Application
FROM node:18-alpine AS build
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Accept the backend API URL as an argument during the build process
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Build the app (Vite will bake the VITE_API_BASE_URL into the static files)
RUN npm run build

# Stage 2: Serve the static files using Nginx
FROM nginx:alpine

# Copy the built React app from Stage 1 to Nginx's web directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port (Cloud Run defaults to 8080, Nginx defaults to 80, but can be overridden)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
