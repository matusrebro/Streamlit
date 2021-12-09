mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
[mapbox]\n\
\n\
" > ~/.streamlit/config.toml