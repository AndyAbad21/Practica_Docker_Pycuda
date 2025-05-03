# Etapa 1: Build Angular app
FROM node:20 AS build

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Mover archivos desde browser/ al root (por culpa del prerender)
RUN cp -r dist/filtros-frontend/browser/* dist/filtros-frontend/

# Etapa 2: Nginx para servir la app
FROM nginx:alpine
RUN rm -rf /usr/share/nginx/html/*
COPY --from=build /app/dist/filtros-frontend /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
