#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include "geometry.hpp"
// Í¨ÓÃ

// Duck ÅäÖÃ
constexpr Material duckMaterial = { {0.0,0.5, 0.1, 0.8}, {0.6, 0.7, 0.8, 125.0},  {1.5, 0.0, 0.0, 0.0} };
constexpr glm::vec3 duckScale = { 1.0f, 1.0f, 1.0f };
constexpr glm::vec3 duckRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 duckTranslation = { 0.0f, 0.0f, 0.0f };
Model duck(duckMaterial);

// Asschercut ÅäÖÃ
constexpr Material asschercutMaterial = { {0.0f, 0.5f, 0.1f, 0.8f}, {0.6f, 0.7f, 0.8f, 125.0f}, {1.5f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 asschercutScale = { 0.5f, 0.5f, 0.5f };
constexpr glm::vec3 asschercutRotation = { 45.0f, 45.0f, 0.0f };
constexpr glm::vec3 asschercutTranslation = { 0.2f, -0.4f, -2.0f };
Model Asschercut(asschercutMaterial);

// Bunny ÅäÖÃ
constexpr Material bunnyMaterial = { {0.9f, 0.1f, 0.0f, 0.0f}, {0.0522876f, 0.156863f, 0.117647f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 bunnyScale = { 0.3f, 0.3f, 0.3f };
constexpr glm::vec3 bunnyRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 bunnyTranslation = { 0.9f, -1.0f, -1.5f };
Model Bunny(bunnyMaterial);

// Dragon ÅäÖÃ
constexpr Material dragonMaterial = { {0.9f, 0.1f, 0.0f, 0.0f}, {0.013072f, 0.078431f, 0.143791f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 dragonScale = { 0.7f, 0.7f, 0.7f };
constexpr glm::vec3 dragonRotation = { 0.0f, 110.0f, 0.0f };
constexpr glm::vec3 dragonTranslation = { -4.15f, -3.0f, -7.0f };
Model Dragon(dragonMaterial);

// Venus ÅäÖÃ
constexpr Material venusMaterial = { {0.0, 10.0, 0.8, 0.0}, {1.0, 1.0, 1.0, 1425.0}, {1.0, 0.0, 0.0, 0.0} };
constexpr glm::vec3 venusScale = { 5.5f, 5.5f, 5.5f };
constexpr glm::vec3 venusRotation = { 0.0f, 0.0f, 0.0f };
constexpr glm::vec3 venusTranslation = { -1.0f, 2.0f, -19.0f };
Model Venus(venusMaterial);

// FudanLogo ÅäÖÃ
constexpr Material fudanlogoMaterial = { {0.7f, 0.3f, 0.0f, 0.0f}, {0.235294f, 0.130719f, 0.078431f, 10.0f}, {1.0f, 0.0f, 0.0f, 0.0f} };
constexpr glm::vec3 fudanlogoScale = { 0.4f, 0.4f, 0.4f };
constexpr glm::vec3 fudanlogoRotation = { 90.0f, 0.0f, 0.0f };
constexpr glm::vec3 fudanlogoTranslation = { -0.5f, 2.0f, -4.0f };
Model FudanLogo(fudanlogoMaterial);