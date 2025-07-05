#pragma once

#include "Sphere.h"
#include "Ray.h"
#include "Light.h"
#include "Triangle.h"
#include "Square.h"

#include <vector>

namespace hlab
{
	// 일반적으로는 헤더파일에서 using namespace std를 사용하지 않습니다.
	// 여기서는 강의 동영상에 녹화되는 코드 길이를 줄이기 위해서 사용하였습니다.
	// 예: std::vector -> vector
	using namespace std;
	using namespace glm;

	class Raytracer
	{
	public:
		int width, height;
		Light light;
		vector<shared_ptr<Object>> objects;

		Raytracer(const int& width, const int& height)
			: width(width), height(height)
		{
			auto sphere1 = make_shared<Sphere>(vec3(0.0f, -0.1f, 4.0f), 1.5f);

			sphere1->amb = vec3(0.1f);
			sphere1->dif = vec3(1.0f, 0.0f, 0.0f);
			sphere1->spec = vec3(1.0f);
			sphere1->alpha = 25.0f;
			sphere1->reflection = 0.5f;
			sphere1->transparency = 0.1f;

			objects.push_back(sphere1);

			auto sphere2 = make_shared<Sphere>(vec3(1.3f, -1.0f, 2.0f), 0.5f);

			sphere2->amb = vec3(0.2f);
			sphere2->dif = vec3(0.0f, 0.0f, 1.0f);
			sphere2->spec = vec3(1.0f);
			sphere2->alpha = 25.0f;
			sphere2->reflection = 0.2f;
			sphere2->transparency = 0.1f;

			objects.push_back(sphere2);

			auto sphere3 = make_shared<Sphere>(vec3(-1.8f, -0.5f, 2.0f), 0.8f);

			sphere3->amb = vec3(1.0f);
			sphere3->dif = vec3(1.0f);
			sphere3->spec = vec3(1.0f);
			sphere3->alpha = 25.0f;
			sphere3->reflection = 0.5f;
			sphere3->transparency = 0.5f;

			objects.push_back(sphere3);

			auto groundTexture = std::make_shared<Texture>("shadertoy_abstract1.jpg");

			auto ground = make_shared<Square>(vec3(-10.0f, -1.5f, 0.0f), vec3(-10.0f, -1.5f, 10.0f), vec3(10.0f, -1.5f, 10.0f), vec3(10.0f, -1.5f, 0.0f),
				vec2(0.0f, 0.0f), vec2(5.0f, 0.0f), vec2(5.0f, 5.0f), vec2(0.0f, 5.0f));

			ground->amb = vec3(0.2f);
			ground->dif = vec3(0.8f);
			ground->spec = vec3(1.0f);
			ground->alpha = 10.0f;
			ground->reflection = 0.5f;
			ground->ambTexture = groundTexture;
			ground->difTexture = groundTexture;

			objects.push_back(ground);

			auto squareTexture = std::make_shared<Texture>("../SaintPetersBasilica/posz_blurred.jpg");
			auto squareTexture2 = std::make_shared<Texture>("../SaintPetersBasilica/negz_blurred.jpg");
			auto squareTexture3 = std::make_shared<Texture>("../SaintPetersBasilica/posy_blurred.jpg");
			auto squareTexture4 = std::make_shared<Texture>("../SaintPetersBasilica/negy_blurred.jpg");
			auto squareTexture5 = std::make_shared<Texture>("../SaintPetersBasilica/posx_blurred.jpg");
			auto squareTexture6 = std::make_shared<Texture>("../SaintPetersBasilica/negx_blurred.jpg");

			auto square = make_shared<Square>(vec3(-60.0f, 30.0f, 30.0f), vec3(60.0f, 30.0f, 30.0f), vec3(60.0f, -30.0f, 30.0f), vec3(-60.0f, -30.0f, 30.0f),
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f));

			auto square2 = make_shared<Square>(
				vec3(-60.0f, -30.0f, -30.0f), vec3(60.0f, -30.0f, -30.0f),
				vec3(60.0f, 30.0f, -30.0f), vec3(-60.0f, 30.0f, -30.0f),
				vec2(0.0f, 1.0f), vec2(1.0f, 1.0f), vec2(1.0f, 0.0f), vec2(0.0f, 0.0f)
			);

			auto square3 = make_shared<Square>(
				vec3(-60.0f, 30.0f, -30.0f), vec3(60.0f, 30.0f, -30.0f),
				vec3(60.0f, 30.0f, 30.0f), vec3(-60.0f, 30.0f, 30.0f),
				vec2(0.0f, 1.0f), vec2(1.0f, 1.0f), vec2(1.0f, 0.0f), vec2(0.0f, 0.0f)
			);


			auto square4 = make_shared<Square>(
				vec3(-60.0f, -30.0f, 30.0f), vec3(60.0f, -30.0f, 30.0f),
				vec3(60.0f, -30.0f, -30.0f), vec3(-60.0f, -30.0f, -30.0f),
				vec2(0.0f, 1.0f), vec2(1.0f, 1.0f), vec2(1.0f, 0.0f), vec2(0.0f, 0.0f)
			);

			auto square5 = make_shared<Square>(
				vec3(-60.0f, 30.0f, -30.0f),  // 좌상
				vec3(-60.0f, 30.0f, 30.0f),   // 우상
				vec3(-60.0f, -30.0f, 30.0f),  // 우하
				vec3(-60.0f, -30.0f, -30.0f), // 좌하
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f),
				vec2(1.0f, 1.0f), vec2(0.0f, 1.0f)
			);


			auto square6 = make_shared<Square>(
				vec3(60.0f, 30.0f, 30.0f),    // 좌상
				vec3(60.0f, 30.0f, -30.0f),   // 우상
				vec3(60.0f, -30.0f, -30.0f),  // 우하
				vec3(60.0f, -30.0f, 30.0f),   // 좌하
				vec2(0.0f, 0.0f), vec2(1.0f, 0.0f),
				vec2(1.0f, 1.0f), vec2(0.0f, 1.0f)
			);


			square->amb = vec3(1.0f);
			square->dif = vec3(0.0f);
			square->spec = vec3(0.0f);
			square->alpha = 50.0f;
			square->reflection = 0.0f;
			square->ambTexture = squareTexture;
			square->difTexture = squareTexture;

			square2->amb = vec3(1.0f);
			square2->dif = vec3(0.0f);
			square2->spec = vec3(0.0f);
			square2->alpha = 50.0f;
			square2->reflection = 0.0f;
			square2->ambTexture = squareTexture2;
			square2->difTexture = squareTexture2;

			square3->amb = vec3(1.0f);
			square3->dif = vec3(0.0f);
			square3->spec = vec3(0.0f);
			square3->alpha = 50.0f;
			square3->reflection = 0.0f;
			square3->ambTexture = squareTexture3;
			square3->difTexture = squareTexture3;

			square4->amb = vec3(1.0f);
			square4->dif = vec3(0.0f);
			square4->spec = vec3(0.0f);
			square4->alpha = 50.0f;
			square4->reflection = 0.0f;
			square4->ambTexture = squareTexture4;
			square4->difTexture = squareTexture4;

			square5->amb = vec3(1.0f);
			square5->dif = vec3(0.0f);
			square5->spec = vec3(0.0f);
			square5->alpha = 50.0f;
			square5->reflection = 0.0f;
			square5->ambTexture = squareTexture5;
			square5->difTexture = squareTexture5;

			square6->amb = vec3(1.0f);
			square6->dif = vec3(0.0f);
			square6->spec = vec3(0.0f);
			square6->alpha = 50.0f;
			square6->reflection = 0.0f;
			square6->ambTexture = squareTexture6;
			square6->difTexture = squareTexture6;

			objects.push_back(square);
			objects.push_back(square2);
			objects.push_back(square3);
			objects.push_back(square4);
			objects.push_back(square5);
			objects.push_back(square6);

			light = Light{ {0.0f, 2.0f, -1.2f} };
		}

		Hit FindClosestCollision(Ray& ray)
		{
			float closestD = 1000.0; // inf
			Hit closestHit = Hit{ -1.0, dvec3(0.0), dvec3(0.0) };

			for (int l = 0; l < objects.size(); l++)
			{
				auto hit = objects[l]->CheckRayCollision(ray);

				if (hit.d >= 0.0f)
				{
					if (hit.d < closestD)
					{
						closestD = hit.d;
						closestHit = hit;
						closestHit.obj = objects[l];

						// 텍스춰 좌표
						closestHit.uv = hit.uv;
					}
				}
			}

			return closestHit;
		}

		// 광선이 물체에 닿으면 그 물체의 색 반환
		vec3 traceRay(Ray& ray, const int recurseLevel)
		{
			if (recurseLevel < 0)
				return vec3(0.0f);

			// Render first hit
			const auto hit = FindClosestCollision(ray);

			if (hit.d >= 0.0f)
			{
				glm::vec3 color(0.0f);

				// Diffuse
				const vec3 dirToLight = glm::normalize(light.pos - hit.point);

				glm::vec3 phongColor(0.0f);

				const float diff = glm::max(dot(hit.normal, dirToLight), 0.0f);

				// Specular
				const vec3 reflectDir = hit.normal * 2.0f * dot(dirToLight, hit.normal) - dirToLight;
				const float specular = glm::pow(glm::max(glm::dot(-ray.dir, reflectDir), 0.0f), hit.obj->alpha);

				Ray shadowRay = { hit.point + dirToLight * 1e-4f, dirToLight };
				bool inShadow = false;

				const auto shadowHit = FindClosestCollision(shadowRay);
				const float lightDistance = glm::length(light.pos - hit.point);

				if (shadowHit.d > 0.0f && shadowHit.d < lightDistance)
				{
					inShadow = true;
				}

				// Ambient (항상 적용)
				if (hit.obj->ambTexture)
				{
					phongColor += hit.obj->amb * hit.obj->ambTexture->SampleLinear(hit.uv);
				}
				else
				{
					phongColor += hit.obj->amb;
				}

				if (inShadow == false)
				{
					// Diffuse & Specular
					const float diff = glm::max(dot(hit.normal, dirToLight), 0.0f);
					const vec3 reflectDir = 2.0f * dot(hit.normal, dirToLight) * hit.normal - dirToLight;
					const float specular = glm::pow(glm::max(glm::dot(-ray.dir, reflectDir), 0.0f), hit.obj->alpha);

					if (hit.obj->difTexture)
					{
						phongColor += diff * hit.obj->dif * hit.obj->difTexture->SampleLinear(hit.uv);
					}
					else
					{
						phongColor += diff * hit.obj->dif;
					}

					phongColor += hit.obj->spec * specular;
				}

				// 물체 기본 색 반영 (반사/투명 제외)
				color += phongColor * (1.0f - hit.obj->reflection - hit.obj->transparency);


				if (hit.obj->reflection)
				{
					const auto reflectedDirection = glm::normalize(2.0f * hit.normal * dot(-ray.dir, hit.normal) + ray.dir);
					Ray reflection_ray{ hit.point + reflectedDirection * 1e-4f, reflectedDirection }; // add a small vector to avoid numerical issue

					color += traceRay(reflection_ray, recurseLevel - 1) * hit.obj->reflection;
				}

				// 참고
				// https://samdriver.xyz/article/refraction-sphere (그림들이 좋아요)
				// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel (오류있음)
				// https://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/reflection_refraction.pdf (슬라이드가 보기 좋지는 않지만 정확해요)
				// 교차 검증 필요 + 고급 기술은 쉽게 인터넷에 정리되어 있지 않음(수학도 공부하면 좋음)
				if (hit.obj->transparency)
				{
					/*
						굴절 유도 이전 사전 지식의 정리
						- 내적은 순서를 교환법칙 성립, 하나에 - 가 있어도, 밖으로 뺄 수 있음
						- a.b의 각 '세타' 는 a.b = cos 세타 가 성립한다

						cos^2 세타 + sin^2 세타 = 1

						sin 세타 = sqrt(1 - (a.b)^2)

						굴절 유도
						- 충돌 지점의 normal 은 n
						- 광선의 방향은 d
						(각기 유닛 벡터)

						굴절 벡터 t (유닛)
						 : -n 과 t의 각도(세타 2) 는 n과 d의 각도(세타1) 과
						 sin 세타 1 / sin 세타 2 의 값이 일정하다는 것이 물리학 적으로 증명됨
						 공기 -> 유리 약 1.5
						 공기 -> 물 (약 1.3)
						 유리 -> 공기 (1 /1.5 : 역수)

						 밀도가 높을수록 해당 비율이 크다
						 (공기는 기체지만, 유리는 고체이므로)

						 밀도가 낮은 곳 -> 높은 곳 일때 해당 비율값이 1보다 크다
						 반대의 경우는 작아진다

					*/

					const float ior = 1.3f; // Index of refraction (유리: 1.5, 물: 1.3)

					float eta; // sinTheta1 / sinTheta2
					vec3 normal;

					// 수치 오류 주의! (point에서 진행 방향으로 약간 띄어야 함)

					if (glm::dot(ray.dir, hit.normal) < 0.0f) // 밖에서 안에서 들어가는 경우 (예: 공기->유리)
					{
						eta = ior;
						normal = hit.normal;
					}
					else // 안에서 밖으로 나가는 경우 (예: 유리->공기)
					{
						eta = 1.0f / ior;
						normal = -hit.normal;
					}

					// a . b = cos 세타
					const float cosTheta1 = -glm::dot(ray.dir, normal);
					const float sinTheta1 = sqrtf(1 - powf(cosTheta1, 2.0f)); // cos^2 + sin^2 = 1
					const float sinTheta2 = sinTheta1 / eta;
					const float cosTheta2 = sqrtf(1 - powf(sinTheta2, 2.0f));

					// m 유도 과정은 간단함 (-d + m = dot(-d,n) * n) -> m = dot(-d,n)*n + d
					const vec3 m = glm::normalize(glm::dot(-ray.dir, normal) * normal + ray.dir);
					const vec3 a = m * sinTheta2;
					const vec3 b = -normal * cosTheta2;
					// 굴절각 t의 x 값은 sinTheta2, y값은 cosTheta2 이므로, -n 과 m을 섞어서 해당 값을 구할 수 있음
					const vec3 refractedDirection = glm::normalize(a + b); // transmission

					Ray refractedRay{ hit.point + refractedDirection * 1e-4f,refractedDirection };
					color += traceRay(refractedRay, recurseLevel - 1) * hit.obj->transparency; // 일부의 색을 가져오는 방식이기에 값을 곱해준다
				}

				return color;
			}

			return vec3(0.0f);
		}

		void Render(std::vector<glm::vec4>& pixels)
		{
			std::fill(pixels.begin(), pixels.end(), vec4(0.0f, 0.0f, 0.0f, 1.0f));

			const vec3 eyePos(0.0f, 0.0f, -1.5f);

#pragma omp parallel for
			for (int j = 0; j < height; j++)
				for (int i = 0; i < width; i++)
				{
					const vec3 pixelPosWorld = TransformScreenToWorld(vec2(i, j));
					Ray pixelRay{ pixelPosWorld, glm::normalize(pixelPosWorld - eyePos) };
					pixels[i + width * j] = vec4(glm::clamp(traceRay(pixelRay, 5), 0.0f, 1.0f), 1.0f);
				}
		}

		vec3 TransformScreenToWorld(vec2 posScreen)
		{
			const float xScale = 2.0f / this->width;
			const float yScale = 2.0f / this->height;
			const float aspect = float(this->width) / this->height;

			// 3차원 공간으로 확장 (z좌표는 0.0)
			return vec3((posScreen.x * xScale - 1.0f) * aspect, -posScreen.y * yScale + 1.0f, 0.0f);
		}
	};
}