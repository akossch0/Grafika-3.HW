//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Schneider Ákos
// Neptun : XYUXUA
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 30;
const int numberOfTracs = 12;
float radiusOfVirus = 2.0f;
const float eps = 0.001f;

//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
Clifford Sinh(Clifford g) { return Clifford(sinhf(g.f), coshf(g.f) * g.d); }
Clifford Cosh(Clifford g) { return Clifford(coshf(g.f), sinf(g.f) * g.d); }
Clifford Tanh(Clifford g) { return Sinh(g) / Cosh(g); }

Clifford radiusFunc(Clifford U, Clifford V, float t) {
	//return Clifford((sinf(2 * t) + 1) / 7.0f, 0) * Sin(U + t) * Sin(V * 10 + t) + (1 - (sinf(t) + 1) / 8.0f);
	return Clifford((sinf(3 * t) + 1) / 7.0f, 0) * 0.75f * Sin(U * 4 + V * 4 + t) + (1 - (sinf(t) + 1) / 8.0f);
}

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinhf(g.f), coshf(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(coshf(g.f), sinhf(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g) {
	return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
typedef Dnum<vec2> Dnum2;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float) M_PI / 180.0f;
		fp = 1; bp = 30;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
													u.y, v.y, w.y, 0,
													u.z, v.z, w.z, 0,
													0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void Animate(float t) { }
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class PurpleTexture : public Texture {
	//---------------------------
public:
	PurpleTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		//
		const vec4 turkis(0.3f, 0.7f, 1, 1), purple(0.6f, 0.5f, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x%2 == 0) ? purple : purple;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class BlueTexture : public Texture {
public:
	BlueTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 light_blue(0, 1, 1, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? light_blue : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};
class BrownTexture : public Texture {
public:
	BrownTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 brown(0.7f, 0.7f, 0.2f, 1), black(0.0f, 0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x%2==0) ? brown : black;
		}
		create(width, height, image, GL_NEAREST);
	}
};
class RedTexture : public Texture {
public:
	RedTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(0.6f, 0.0f, 0.0f, 1), white(0.4, 0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x % 2 == 0) ? red : white;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class YellowTexture : public Texture {
public:
	YellowTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), green(0.5f, 1, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : green;
		}
		create(width, height, image, GL_NEAREST);
	}
};
class GreyTexture : public Texture {
public:
	GreyTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 grey1(0.65f, 0.65f, 0.65f, 1), grey2(0.1f, 0.1f, 0.1f, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x%2 ==0) ? grey2 : grey1;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4				 MVP, M, Minv, V, P;
	Material*			 material;
	std::vector<Light>	lights;
	Texture*			texture;
	vec3				wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
	
	unsigned int vao, vbo;        // vertex array object
public:
	
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual VertexData GenVertexData(float u, float v, float t_end) = 0;

	virtual void create(int N = tessellationLevel, int M = tessellationLevel, float t_end = 0) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N, t_end));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N, t_end));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};



//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }

	VertexData GenVertexData(float u, float v, float t_end) {
		VertexData vd;
		vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};
//---------------------------
struct CoronaBody : public ParamSurface {
//---------------------------
	CoronaBody(float t = 0) { create(tessellationLevel, tessellationLevel, t); }

	void eval(Clifford& U, Clifford& V, Clifford& X, Clifford& Y, Clifford& Z, Clifford& rad) {
		X = rad * Cos(U) * Sin(V);
		Y = rad * Sin(U) * Sin(V);
		Z = rad * Cos(V);
	}

	VertexData GenVertexData(float u, float v, float t_end) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Clifford X, Y, Z;
		Clifford U(u * 2.0f * (float)M_PI, 1), V(v * (float)M_PI, 0);		
		Clifford R = radiusFunc(U, V, t_end);
		eval(U, V, X, Y, Z, R);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		R = radiusFunc(U, V, t_end);
		eval(U, V, X, Y, Z, R);
		vec3 drdV(X.d, Y.d, Z.d);
		vtxData.normal = normalize(cross(drdU, drdV));
		return vtxData;		
	}
};
class Tractricoid : public ParamSurface {
	float height;
public:
	Tractricoid(float h) : height(h) { create(); }

	void eval(Clifford& U, Clifford& V, Clifford& X, Clifford& Y, Clifford& Z) {
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
	}

	VertexData GenVertexData(float u, float v, float t_end) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Clifford X, Y, Z;
		Clifford U(u * height, 1), V(v * 2.0f * M_PI, 0);
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		eval(U, V, X, Y, Z);
		vec3 drdV(X.d, Y.d, Z.d);
		vtxData.normal = normalize(cross(drdU, drdV));
		return vtxData;
	}
};
class CylinderZ : public ParamSurface {
public:
	CylinderZ() { create(); }

	VertexData GenVertexData(float u, float v, float t_end) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Clifford X, Y, Z;
		Clifford U(u * 2.0f * M_PI, 1), V(v * 2.0f - 1.0f, 0);
		X = Cos(U);
		Y = Sin(U);
		Z = V;
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		X = Cos(U);
		Y = Sin(U);
		Z = V;
		vec3 drdV(X.d, Y.d, Z.d);
		vtxData.normal = normalize(cross(drdU, drdV));
		return vtxData;
	}
};

struct Triangle {
	vec3 v1, v2, v3, normal;
	//vec2 texcoord = vec2(0, 0);
	Triangle(vec3 _v1, vec3 _v2, vec3 _v3): v1(_v1), v2(_v2), v3(_v3){
		normal = normalize(cross((v2 - v1), (v3 - v2)));
	}
};


class AntiVirusBody : public Geometry {
	std::vector<Triangle> triangs;
	std::vector<VertexData> vtxData;
	float height;
public:
	AntiVirusBody(float _height, float t = 0) { 
		create(tessellationLevel, tessellationLevel, t);
		height = _height;
	}

	std::vector<VertexData> triangsToVertexData(Triangle& tri) {
		std::vector<VertexData> vds;
		VertexData vd1, vd2, vd3;
		vd1.position = tri.v1;
		vd1.normal = tri.normal;
		vd1.texcoord = vec2(tri.v1.x, tri.v1.y);
		vd2.position = tri.v2;
		vd2.normal = tri.normal;
		vd2.texcoord = vec2(tri.v2.x, tri.v2.y);
		vd3.position = tri.v3;
		vd3.normal = tri.normal;
		vd3.texcoord = vec2(tri.v3.x, tri.v3.y);
		vds.push_back(vd1);
		vds.push_back(vd2);
		vds.push_back(vd3);

		return vds;
	}

	VertexData GenVertexData(float u, float v, float t_end) { 
		VertexData vd;
		return vd;
	}

	Triangle Halfway(Triangle tri) {
		return Triangle((tri.v1 + tri.v2) / 2, (tri.v2 + tri.v3) / 2, (tri.v3 + tri.v1) / 2);
	}

	void recVertex(std::vector<Triangle>& tris, std::vector<Triangle>& append, float time) {
		float t_scale = fabs(sinf(3 * time)/1.7f) + 1;
		for (int i = 0; i < tris.size(); i++) {
			Triangle t(Halfway(tris[i]));
			vec3 center = (t.v1 + t.v2 + t.v3) / 3;
			vec3 normal = t.normal;
			float scale = sqrtf(length(t.v2 - t.v1) * length(t.v2 - t.v1) - (length(t.v2 - t.v1) / 2) * (length(t.v2 - t.v1) / 2));
			vec3 top = (center + normal * scale) * t_scale;
			
			append.push_back(Triangle(t.v1, t.v2, top));
			append.push_back(Triangle(top, t.v2, t.v3));
			append.push_back(Triangle(t.v3, t.v1, top));
		}
		tris.insert(tris.end(), append.begin(), append.end());
	}

	void create(int N = tessellationLevel, int M = tessellationLevel, float t_end = 0) {

		vec3 p1(1, 1, 1);
		vec3 p2(1, -1, -1);
		vec3 p3(-1, 1, -1);
		vec3 p4(-1, -1, 1);

		triangs.push_back(Triangle(p1, p2, p3));
		triangs.push_back(Triangle(p1, p4, p2));
		triangs.push_back(Triangle(p1, p3, p4));
		triangs.push_back(Triangle(p3, p2, p4));
		
		std::vector<Triangle> app1;
		std::vector<Triangle> app2;
		recVertex(triangs, app1, t_end);
		recVertex(app1, app2, t_end);
		triangs.insert(triangs.end(), app2.begin(), app2.end());

		for (int i = 0; i < triangs.size(); i++) {
			std::vector<VertexData> tmp = triangsToVertexData(triangs[i]);
			vtxData.insert(vtxData.end(), tmp.begin(), tmp.end());
		}

		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vtxData.size());
	}
	
};

//---------------------------
struct Object {
//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	std::vector<Object*> children;
public:
	Object() {}
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { 
		rotationAngle = 0.2f * sinf(tend);
	}
};
struct TracObj : public Object {
	float u, v;
	vec3 i, j, dir, trans;
	mat4 transform;
	mat4 invtransform;

	TracObj(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry, float _u, float _v): Object() {
		shader = _shader;
		material = _material;
		texture = _texture;
		geometry = _geometry;
		u = _u;
		v = _v;
		scale = vec3(0.14f, 0.14f, 0.1f);
	}

	void Draw(RenderState state) {
		state.M = ScaleMatrix(scale)* transform * state.M;
		state.Minv = state.Minv * invtransform * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void calculateNormal(float u, float vn, float t_end) {
		Clifford U(u * 2.0f * M_PI, 1), V(v * M_PI, 0);
		Clifford R = radiusFunc(U, V, t_end);
		Clifford X = R * Cos(U) * Sin(V);
		Clifford Y = R * Sin(U) * Sin(V);
		Clifford Z = R * Cos(V);
		trans = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		X = R * Cos(U) * Sin(V);
		Y = R * Sin(U) * Sin(V);
		Z = R * Cos(V);
		vec3 drdV(X.d, Y.d, Z.d);
		i = drdV;
		j = drdU;
		dir = normalize(cross(drdU, drdV));
		trans = vec3(X.f, Y.f, Z.f) - dir/5;
	}

	void Animate(float tstart, float tend) {
		calculateNormal(u, v, tend);

		transform = mat4(i.x, i.y, i.z, 0.0f,
						j.x, j.y, j.z, 0.0f,
						dir.x, dir.y, dir.z, 0.0f,
						trans.x, trans.y, trans.z, 1);

		invtransform = mat4(i.x, j.x, dir.x, 0.0f,
							i.y, j.y, dir.y, 0.0f,
							i.z, j.z, dir.z, 0.0f,
							-trans.x, -trans.y, -trans.z, 1);
	}
};


struct CoronaVirus : public Object {	

	CoronaVirus(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		Object(_shader, _material, _texture, _geometry) { }

	void Animate(float tstart, float tend) {
		rotationAngle = tend * 0.9f;
		translation = vec3(cosf(tend / 1.4) * 1.5, 3*cosf(tend), -fabs(2*cosf(tend))+2);
		//rotationAxis = vec3(sinf(4 * tend), cosf(3 * tend), cosf(tend));
		delete geometry;
		geometry = new CoronaBody(tend);
	}

	void populateChildren(Shader* phong, Material* mat, Texture* text, Geometry* g) {
		
		for (int i = 0; i < numberOfTracs + 1; i++) {
			int num = (int)((float)numberOfTracs * (sinf((float)i / numberOfTracs * M_PI)));
			for (int j = 0; j <= num; j++) {
				float u = (float)j / (num + 1);
				float v = (float)i / numberOfTracs + eps*10;

				Object* trac = new TracObj(phong, mat, text, g, u, v);
				children.push_back(trac);
			}
		}
	}
	void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
		for (Object* child : children) child->Draw(state);
	}

};

struct Room : public Object {

	Room(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		Object(_shader, _material, _texture, _geometry) {}

	void Animate(float tstart, float tend) {}
};

struct AntiVirus : public Object {
	AntiVirus(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		Object(_shader, _material, _texture, _geometry) { }

	void Animate(float tstart, float tend) {
		rotationAngle = tend * 0.5f;
		translation = vec3(3*sinf(tend/2), 2*sinf(tend*2), -fabs(8 * sin(tend)));
		rotationAxis = vec3(sinf(tend), cosf(tend), cosf(tend));
		delete geometry;
		geometry = new AntiVirusBody(1.0f, tend);
	}
};


//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 10;


		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);
		Texture* blue10x20 = new BlueTexture(10, 20);
		Texture* yellow5x5 = new YellowTexture(5, 5);
		Texture* grey1x1 = new GreyTexture(1, 1);
		Texture* grey20x20 = new GreyTexture(20, 20);
		Texture* purple = new PurpleTexture(10, 10);
		Texture* brown50x50 = new BrownTexture(50,50);
		Texture* red4x4 = new RedTexture(4, 4);

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* coronaBody = new CoronaBody();
		Geometry* tractricoid = new Tractricoid(3.0f);
		Geometry* cylinderZ = new CylinderZ();
		Geometry* antiVirusBody = new AntiVirusBody(1.0f);

		// Create objects by setting up their vertex data on the GPU
		CoronaVirus* corona = new CoronaVirus(phongShader, material0, brown50x50, coronaBody);
		corona->translation = vec3(3, 0, 0);
		corona->rotationAxis = vec3(1, 1, 1);
		corona->scale = vec3(1, 1, 1);
		corona->populateChildren(phongShader,material0, red4x4, tractricoid);
		objects.insert(objects.end(), corona->children.begin(), corona->children.end());
		objects.push_back(corona);

		Object* room1 = new Object(phongShader, material1, grey20x20, cylinderZ);
		room1->translation = vec3(0, 0, 0);
		room1->rotationAxis = vec3(0, 0, 1);
		room1->scale = vec3(10, 10, 10);
		objects.push_back(room1);

		Object* room2 = new Room(phongShader, material1, grey1x1, sphere);
		room2->translation = vec3(0, 0, -9.9);
		room2->rotationAxis = vec3(0, 0, 1);
		room2->scale = vec3(10, 10, 0.01f);
		objects.push_back(room2);

		Object* anti = new AntiVirus(phongShader, material0, purple, antiVirusBody);
		anti->translation = vec3(0, 0, 0);
		anti->rotationAxis = vec3(1, 1, 1);
		anti->scale = vec3(1, 1, 1);
		objects.push_back(anti);


		// Camera
		camera.wEye = vec3(0, 0, 6);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(0, 0, 20, 5);	// ideal point -> directional light source
		lights[0].La = vec3(0.3f, 0.3f, 0.3f);
		lights[0].Le = vec3(3.5f, 3.5f, 3.5f);
		
		/*lights[1].wLightPos = vec4(0, 0, 20, 1);	// ideal point -> directional light source
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(0, 0, 20, 1);	// ideal point -> directional light source
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);*/
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera.Animate(tend);
		for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}