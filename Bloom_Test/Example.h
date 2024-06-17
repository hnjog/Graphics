#pragma once

#include <windows.h>
#include <memory>
#include <iostream>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <chrono>
#include <algorithm>


// 수학의 4차원 벡터(vector)를 의미합니다. std::vector와는 다릅니다.
struct Vec4
{
	float v[4];
};

struct Vec2
{
	float v[2];
};

struct Vertex
{
	Vec4 pos;
	Vec2 uv;
};

class Image
{
public:
	int width = 0, height = 0, channels = 0;
	std::vector<Vec4> pixels; // 이미지 처리할 때는 색을 float에 저장하는 것이 더 정밀

	void ReadFromFile(const char* filename);
	void WritePNG(const char* filename);
	Vec4& GetPixel(int i, int j);
	void BoxBlur5();
	void GaussianBlur5();
	void Bloom(const float& th, const int& numRepeat, const float& weight = 1.0f);
};

class Example
{
public:
	Example(HWND window, int width, int height)
	{
		// 이미지 읽어들이기
		image.ReadFromFile("image_1_360.jpg"); // 컴퓨터 속도가 느리다면 "image_1_360.jpg" 사용

		// 시간 측정
		const auto start_time = std::chrono::high_resolution_clock::now();

		/*
		* 이미지 컨벌루션 참고 자료들
		* https://en.wikipedia.org/wiki/Kernel_(image_processing) // 마지막 괄호 주의
		* https://en.wikipedia.org/wiki/Convolution
		* https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411
		* https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
		*/

		/*
		* 여기서 사용하는 방법은 이해하기 더 쉬운 Separable convolution 입니다.
		*/

		/*for(int i = 0; i < 100; i++)
			image.BoxBlur5();*/
		
		for (int i = 0; i < 100; i++)
			image.GaussianBlur5();

		//image.Bloom(0.3f, 1000, 1.0f);// image_1

		// Image 밝게
		// pixel 내부의 rgb 값을 증가 (1에 가까울 수록 밝아진다)
		//for (auto& pixel : image.pixels)
		//{
		//	pixel.v[0] *= 1.2f;
		//	pixel.v[1] *= 1.2f;
		//	pixel.v[2] *= 1.2f;
		//}

		// '밝게' 하였을 때, 색이 일부 변하지 않는 경우를 수정하기
		// 미세한 값을 더하여 0.0f가 아니게 한다
		//for (auto& pixel : image.pixels)
		//{
		//	pixel.v[0] += 1e-2f;
		//	pixel.v[1] += 1e-2f;
		//	pixel.v[2] += 1e-2f;
		//}

		const auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;

		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0 << " sec" << std::endl;

		image.WritePNG("result.png");

		// this->canvasWidth = image.width
		// this->canvasHeight = image.height
		Initialize(window, width, height, image.width, image.height);
	}

	// https://docs.microsoft.com/en-us/windows/win32/direct3d11/how-to--compile-a-shader
	void InitShaders()
	{
		ID3DBlob* vertexBlob = nullptr;
		ID3DBlob* pixelBlob = nullptr;
		ID3DBlob* errorBlob = nullptr;

		if (FAILED(D3DCompileFromFile(L"VS.hlsl", 0, 0, "main", "vs_5_0", 0, 0, &vertexBlob, &errorBlob)))
		{
			if (errorBlob) {
				std::cout << "Vertex shader compile error\n" << (char*)errorBlob->GetBufferPointer() << std::endl;
			}
		}

		if (FAILED(D3DCompileFromFile(L"PS.hlsl", 0, 0, "main", "ps_5_0", 0, 0, &pixelBlob, &errorBlob)))
		{
			if (errorBlob) {
				std::cout << "Pixel shader compile error\n" << (char*)errorBlob->GetBufferPointer() << std::endl;
			}
		}

		device->CreateVertexShader(vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), NULL, &vertexShader);
		device->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), NULL, &pixelShader);

		// Create the input layout object
		D3D11_INPUT_ELEMENT_DESC ied[] =
		{
			{"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
			{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
		};

		device->CreateInputLayout(ied, 2, vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(), &layout);
		deviceContext->IASetInputLayout(layout);
	}

	void Initialize(HWND window, int width, int height, int canvasWidth, int canvasHeight)
	{
		this->canvasWidth = canvasWidth;
		this->canvasHeight = canvasHeight;

		DXGI_SWAP_CHAIN_DESC swapChainDesc;
		ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));
		swapChainDesc.BufferDesc.Width = width;                   // set the back buffer width
		swapChainDesc.BufferDesc.Height = height;                 // set the back buffer height
		swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;    // use 32-bit color
		swapChainDesc.BufferCount = 2;                                   // one back buffer
		swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
		swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
		swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;     // how swap chain is to be used
		swapChainDesc.OutputWindow = window;                               // the window to be used
		swapChainDesc.SampleDesc.Count = 1;                              // how many multisamples
		swapChainDesc.Windowed = TRUE;                                   // windowed/full-screen mode
		swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;    // allow full-screen switching
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

		UINT createDeviceFlags = 0;
		//createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;

		const D3D_FEATURE_LEVEL featureLevelArray[1] = { D3D_FEATURE_LEVEL_11_0 };
		if (FAILED(D3D11CreateDeviceAndSwapChain(NULL,
			D3D_DRIVER_TYPE_HARDWARE,
			NULL,
			createDeviceFlags,
			featureLevelArray,
			1,
			D3D11_SDK_VERSION,
			&swapChainDesc,
			&swapChain,
			&device,
			NULL,
			&deviceContext)))
		{
			std::cout << "D3D11CreateDeviceAndSwapChain() error" << std::endl;
		}

		// CreateRenderTarget
		ID3D11Texture2D* pBackBuffer;
		swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
		if (pBackBuffer)
		{
			device->CreateRenderTargetView(pBackBuffer, NULL, &renderTargetView);
			pBackBuffer->Release();
		}
		else
		{
			std::cout << "CreateRenderTargetView() error" << std::endl;
			exit(-1);
		}

		// Set the viewport
		ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		viewport.Width = float(width);
		viewport.Height = float(height);
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;//Note: important for depth buffering
		deviceContext->RSSetViewports(1, &viewport);

		InitShaders();

		// Create texture and rendertarget
		D3D11_SAMPLER_DESC sampDesc;
		ZeroMemory(&sampDesc, sizeof(sampDesc));
		sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT; // D3D11_FILTER_MIN_MAG_MIP_LINEAR
		sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		sampDesc.MinLOD = 0;
		sampDesc.MaxLOD = D3D11_FLOAT32_MAX;

		//Create the Sample State
		device->CreateSamplerState(&sampDesc, &colorSampler);

		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(textureDesc));
		textureDesc.MipLevels = textureDesc.ArraySize = 1;
		textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.Usage = D3D11_USAGE_DYNAMIC;
		textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		textureDesc.MiscFlags = 0;
		textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		textureDesc.Width = canvasWidth;
		textureDesc.Height = canvasHeight;

		device->CreateTexture2D(&textureDesc, nullptr, &canvasTexture);

		if (canvasTexture)
		{
			device->CreateShaderResourceView(canvasTexture, nullptr, &canvasTextureView);

			D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
			renderTargetViewDesc.Format = textureDesc.Format;
			renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			renderTargetViewDesc.Texture2D.MipSlice = 0;

			device->CreateRenderTargetView(canvasTexture, &renderTargetViewDesc, &canvasRenderTargetView);
		}
		else
		{
			std::cout << "CreateRenderTargetView() error" << std::endl;
		}

		// Create a vertex buffer
		{
			const std::vector<Vertex> vertices =
			{
				{ { -1.0f, -1.0f, 0.0f, 1.0f }, { 0.f, 1.f },},
				{ {  1.0f, -1.0f, 0.0f, 1.0f }, { 1.f, 1.f },},
				{ {  1.0f,  1.0f, 0.0f, 1.0f }, { 1.f, 0.f },},
				{ { -1.0f,  1.0f, 0.0f, 1.0f }, { 0.f, 0.f },},
			};

			D3D11_BUFFER_DESC bufferDesc;
			ZeroMemory(&bufferDesc, sizeof(bufferDesc));
			bufferDesc.Usage = D3D11_USAGE_DYNAMIC;                // write access access by CPU and GPU
			bufferDesc.ByteWidth = UINT(sizeof(Vertex) * vertices.size());             // size is the VERTEX struct * 3
			bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;       // use as a vertex buffer
			bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;    // allow CPU to write in buffer
			bufferDesc.StructureByteStride = sizeof(Vertex);

			D3D11_SUBRESOURCE_DATA vertexBufferData = { 0, };
			vertexBufferData.pSysMem = vertices.data();
			vertexBufferData.SysMemPitch = 0;
			vertexBufferData.SysMemSlicePitch = 0;

			const HRESULT hr = device->CreateBuffer(&bufferDesc, &vertexBufferData, &vertexBuffer);
			if (FAILED(hr)) {
				std::cout << "CreateBuffer() failed. " << std::hex << hr << std::endl;
			};
		}

		// Create an index buffer
		{
			const std::vector<uint16_t> indices =
			{
				3,1,0,
				2,1,3,
			};

			indexCount = UINT(indices.size());

			D3D11_BUFFER_DESC bufferDesc = {};
			bufferDesc.Usage = D3D11_USAGE_DYNAMIC;                // write access access by CPU and GPU
			bufferDesc.ByteWidth = UINT(sizeof(uint16_t) * indices.size());
			bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;       // use as a vertex buffer
			bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;    // allow CPU to write in buffer
			bufferDesc.StructureByteStride = sizeof(uint16_t);

			D3D11_SUBRESOURCE_DATA indexBufferData = { 0 };
			indexBufferData.pSysMem = indices.data();
			indexBufferData.SysMemPitch = 0;
			indexBufferData.SysMemSlicePitch = 0;

			device->CreateBuffer(&bufferDesc, &indexBufferData, &indexBuffer);
		}
	}

	void Update()
	{
		/* 앞 예제
		std::vector<Vec4> pixels(canvasWidth * canvasHeight, Vec4{0.8f, 0.8f, 0.8f, 1.0f});
		pixels[0 + canvasWidth * 0] = Vec4{ 1.0f, 0.0f, 0.0f, 1.0f };
		pixels[1 + canvasWidth * 0] = Vec4{ 1.0f, 1.0f, 0.0f, 1.0f };
		*/

		//image.Blur(7);

		//image.GaussianBlur5();

		// 점점 어두워지는 연출
		//for (auto& pixel : image.pixels)
		//{
		//	pixel.v[0] = std::clamp(pixel.v[0] * 0.9f, 0.0f, 1.0f);
		//	pixel.v[1] = std::clamp(pixel.v[1] * 0.9f, 0.0f, 1.0f);
		//	pixel.v[2] = std::clamp(pixel.v[2] * 0.9f, 0.0f, 1.0f);
		//}

		// 반반
		// 2 중 for문으로 index를 체크할 때 할 수 있는 방식
		//for (int i = 0; i < image.height; i++)
		//{
		//	for (int j = 0; j < image.width; j++)
		//	{
		//		const int idx = j + image.width * i;

		//		if (j > image.width / 2)
		//		{
		//			// 너무 밝아도 약간 '선'같은게 보이는 경우가 있는데
		//			// 맨 처음 불러올 때 아예 0.0 인 경우가 존재하기에 발생
		//			image.pixels[idx].v[0] = std::clamp(image.pixels[idx].v[0] * 1.1f, 0.0f, 1.0f);
		//			image.pixels[idx].v[1] = std::clamp(image.pixels[idx].v[1] * 1.1f, 0.0f, 1.0f);
		//			image.pixels[idx].v[2] = std::clamp(image.pixels[idx].v[2] * 1.1f, 0.0f, 1.0f);
		//		}
		//		else
		//		{
		//			image.pixels[idx].v[0] = std::clamp(image.pixels[idx].v[0] * 0.9f, 0.0f, 1.0f);
		//			image.pixels[idx].v[1] = std::clamp(image.pixels[idx].v[1] * 0.9f, 0.0f, 1.0f);
		//			image.pixels[idx].v[2] = std::clamp(image.pixels[idx].v[2] * 0.9f, 0.0f, 1.0f);
		//		}
		//		
		//	}
		//}

		// 여담
		// 메모리 매핑
		// map : 파일 or 디바이스 를 프로세스의 '가상 주소 공간'에 매핑
		// (메모리 매핑 file I/O를 통해 수행)
		// unmap : 매핑된 메모리 영역을 해제
		// 
		// 가상 메모리
		// map : 가상 메모리 시스템에서, 가상 주소를 '물리적 메모리 주소'에 매핑 (page Table로 관리)
		// unmap : 가상 주소와 물리 주소간 매핑 해제

		// 이미지의 내용을 GPU 메모리로 복사
		// CPU의 데이터(메모리 : RAM) 을 GPU에 보내는 방식
		// GPU의 메모리는 VRAM으로 '그래픽카드' 내부에 존재한다
		D3D11_MAPPED_SUBRESOURCE ms;
		// GPU 리소스를 CPU가 접근할 수 있도록 메모리 매핑
		deviceContext->Map(canvasTexture, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms);
		//memcpy(ms.pData, pixels.data(), pixels.size() * sizeof(Vec4));
		// RAM -> VRAM으로 복사하기 위하 memcpy 사용
		memcpy(ms.pData, image.pixels.data(), image.pixels.size() * sizeof(Vec4));
		// 매핑 해제하여 다시 GPU가 접근할 수 있도록 함
		deviceContext->Unmap(canvasTexture, NULL);
	}

	void Render()
	{
		float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
		deviceContext->RSSetViewports(1, &viewport);
		deviceContext->OMSetRenderTargets(1, &renderTargetView, nullptr);
		deviceContext->ClearRenderTargetView(renderTargetView, clearColor);

		// set the shader objects
		deviceContext->VSSetShader(vertexShader, 0, 0);
		deviceContext->PSSetShader(pixelShader, 0, 0);

		// select which vertex buffer to display
		UINT stride = sizeof(Vertex);
		UINT offset = 0;
		deviceContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
		deviceContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R16_UINT, 0);

		// https://github.com/Microsoft/DirectXTK/wiki/Getting-Started
		// https://github.com/microsoft/Xbox-ATG-Samples/tree/main/PCSamples/IntroGraphics/SimpleTexturePC
		deviceContext->PSSetSamplers(0, 1, &colorSampler);//TODO: samplers to array
		deviceContext->PSSetShaderResources(0, 1, &canvasTextureView);
		deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		deviceContext->DrawIndexed(indexCount, 0, 0);
	}

	void Clean()
	{
		if (layout) { layout->Release(); layout = NULL; }
		if (vertexShader) { vertexShader->Release(); vertexShader = NULL; }
		if (pixelShader) { pixelShader->Release(); pixelShader = NULL; }
		if (vertexBuffer) { vertexBuffer->Release(); vertexBuffer = NULL; }
		if (indexBuffer) { indexBuffer->Release(); indexBuffer = NULL; }
		if (canvasTexture) { canvasTexture->Release(); canvasTexture = NULL; }
		if (canvasTextureView) { canvasTextureView->Release(); canvasTextureView = NULL; }
		if (canvasRenderTargetView) { canvasRenderTargetView->Release(); canvasRenderTargetView = NULL; }
		if (colorSampler) { colorSampler->Release(); colorSampler = NULL; }
		if (renderTargetView) { renderTargetView->Release(); renderTargetView = NULL; }
		if (swapChain) { swapChain->Release(); swapChain = NULL; }
		if (deviceContext) { deviceContext->Release(); deviceContext = NULL; }
		if (device) { device->Release(); device = NULL; }
	}

public:
	// 이름 짓기 규칙
	// https://github.com/Microsoft/DirectXTK/wiki/Implementation#naming-conventions
	ID3D11Device* device;
	ID3D11DeviceContext* deviceContext;
	IDXGISwapChain* swapChain;
	D3D11_VIEWPORT viewport;
	ID3D11RenderTargetView* renderTargetView;
	ID3D11VertexShader* vertexShader;
	ID3D11PixelShader* pixelShader;
	ID3D11InputLayout* layout;

	ID3D11Buffer* vertexBuffer = nullptr;
	ID3D11Buffer* indexBuffer = nullptr;
	ID3D11Texture2D* canvasTexture = nullptr;
	ID3D11ShaderResourceView* canvasTextureView = nullptr;
	ID3D11RenderTargetView* canvasRenderTargetView = nullptr;
	ID3D11SamplerState* colorSampler;
	UINT indexCount;

	int canvasWidth, canvasHeight;
	Image image; //std::vector<Vec4> pixels;
};