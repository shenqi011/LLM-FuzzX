import axios from "axios";

const instance = axios.create({
    baseURL: process.env.NODE_ENV === 'development' 
        ? 'http://localhost:5000/api'  // 开发环境直接访问后端
        : '/api',  // 生产环境使用代理
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    }
});

export default instance; 
