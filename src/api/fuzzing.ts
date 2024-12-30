import axios from "./index";

// 更新基础URL
axios.defaults.baseURL = "http://localhost:5000/api";

axios.interceptors.request.use(request => {
    console.log('Starting Request:', request);
    return request;
});

axios.interceptors.response.use(response => {
    console.log('Response:', response);
    return response;
}, error => {
    console.error('Response Error:', error);
    return Promise.reject(error);
});

// 获取可用的问题文件列表
export const getQuestionFiles = async () => {
    try {
        const response = await axios.get<{ files: string[] }>('/question-files', {
            headers: {
                'Access-Control-Allow-Origin': '*'
            }
        });
        console.log('Response:', response);  // 调试日志
        return response.data.files;
    } catch (error) {
        console.error('Error fetching question files:', error);  // 调试日志
        throw new Error('Failed to get question files');
    }
};

