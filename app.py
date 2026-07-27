import pandas as pd
import numpy as np
import json
import math
import os
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from github import Github

try:
    from pymongo import MongoClient
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb+srv://siddardhvanguri_db_user:PTix0CddP9ViH6eQ@cluster0.vraf2ru.mongodb.net/electrical_machines_db?retryWrites=true&w=majority")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["electrical_machines_db"]
    observations_collection = db["observations"]
    users_collection = db["users"]
    improvements_collection = db["improvements"]
    
    # Initialize default admin if not exists
    if users_collection.count_documents({"username": "admin"}) == 0:
        users_collection.insert_one({"username": "admin", "password": "password", "role": "admin"})
    # Initialize default student if not exists
    if users_collection.count_documents({"username": "student"}) == 0:
        users_collection.insert_one({"username": "student", "password": "password", "role": "student"})
        
except ImportError:
    MongoClient = None
    observations_collection = None
    users_collection = None
    improvements_collection = None

app = Flask(__name__)
CORS(app)

# Helper function to serialize Random Forest Regressor
def serialize_tree(tree):
    def recurse(node):
        if tree.feature[node] != -2:
            return {
                "type": "split",
                "feature_idx": int(tree.feature[node]),
                "threshold": float(tree.threshold[node]),
                "left": recurse(tree.children_left[node]),
                "right": recurse(tree.children_right[node])
            }
        else:
            return {
                "type": "leaf",
                "value": float(tree.value[node][0][0])
            }
    return recurse(0)

# Helper function to train and serialize models for a machine
def train_and_serialize_models(df, input_cols, output_cols):
    X = df[input_cols].values
    Y = df[output_cols].values
    num_outputs = Y.shape[1]
    
    # Scale features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # 1. Polynomial Regression (Degree 2)
    poly_feat = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly_feat.fit_transform(X)
    
    poly_coefs = []
    poly_r2 = []
    for i in range(num_outputs):
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_poly, Y[:, i])
        poly_coefs.append(lr.coef_.tolist())
        poly_r2.append(float(lr.score(X_poly, Y[:, i])))
        
    # 2. Random Forest Regressor
    rf_serialized = []
    rf_r2 = []
    for i in range(num_outputs):
        rf = RandomForestRegressor(n_estimators=4, max_depth=3, random_state=42)
        rf.fit(X, Y[:, i])
        rf_r2.append(float(rf.score(X, Y[:, i])))
        serialized_trees = [serialize_tree(dt.tree_) for dt in rf.estimators_]
        rf_serialized.append(serialized_trees)
        
    # 3. Neural Network (MLP Regressor)
    mlp_serialized = []
    mlp_r2 = []
    for i in range(num_outputs):
        mlp = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1500, random_state=42)
        mlp.fit(X_scaled, Y[:, i])
        mlp_r2.append(float(mlp.score(X_scaled, Y[:, i])))
        mlp_serialized.append({
            "W1": mlp.coefs_[0].tolist(),
            "b1": mlp.intercepts_[0].tolist(),
            "W2": mlp.coefs_[1].tolist(),
            "b2": mlp.intercepts_[1].tolist()
        })
        
    return {
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "models": {
            "poly": {
                "coefs": poly_coefs,
                "r2": poly_r2
            },
            "rf": {
                "forests": rf_serialized,
                "r2": rf_r2
            },
            "mlp": {
                "networks": mlp_serialized,
                "r2": mlp_r2
            }
        }
    }

# -------------------------------------------------------------
# PART 1: INDUCTION MOTOR MODEL PIPELINE
# -------------------------------------------------------------

# Dataset 1: 415V Load Test (Star Connected, R_drum = 0.12605 m)
data_415v = {
    "V_line": [415.0, 430.0, 430.0, 435.0, 430.0, 430.0],
    "I_line": [3.2, 3.4, 3.7, 3.9, 4.3, 4.6],
    "P_in": [300.0, 480.0, 1280.0, 1640.0, 2120.0, 2480.0],
    "S1": [0.0, 0.0, 0.2, 0.6, 1.4, 2.5],
    "S2": [0.0, 1.0, 5.0, 6.6, 9.4, 12.3],
    "S1_S2": [0.0, 1.0, 4.8, 6.0, 8.0, 9.8],
    "Speed": [1497.0, 1495.0, 1484.0, 1478.0, 1476.0, 1465.0],
    "T_shaft": [0.0, 1.24, 5.94, 7.42, 9.90, 12.12],
    "P_out": [0.0, 194.13, 923.10, 1148.44, 1530.20, 1854.40],
    "eff": [0.0, 40.45, 72.12, 70.02, 72.18, 74.77],
    "pf": [0.130, 0.190, 0.464, 0.558, 0.661, 0.723],
    "slip": [0.20, 0.33, 1.07, 1.47, 1.60, 2.33],
    "R_drum": 0.12605
}

# Dataset 2: 220V Load Test (Star Connected, R_drum = 0.097 m)
data_220v = {
    "V_line": [220.0, 220.0, 220.0, 220.0, 220.0, 220.0, 220.0, 220.0, 220.0],
    "I_line": [4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4],
    "P_in": [160.0, 280.0, 440.0, 600.0, 640.0, 680.0, 700.0, 920.0, 1000.0],
    "S1": [0.0, 1.7, 2.5, 2.8, 3.7, 3.93, 4.3, 4.6, 4.8],
    "S2": [0.0, 0.2, 0.3, 0.33, 0.4, 0.4, 0.33, 0.3, 0.3],
    "S1_S2": [0.0, 1.5, 2.2, 2.47, 3.3, 3.53, 3.97, 4.3, 4.5],
    "Speed": [1500.0, 1491.0, 1485.0, 1480.0, 1477.0, 1472.0, 1467.0, 1464.0, 1461.0],
    "T_shaft": [0.0, 1.428, 2.090, 2.350, 3.140, 3.360, 3.781, 4.095, 4.286],
    "P_out": [0.0, 222.60, 325.01, 364.20, 486.10, 518.20, 580.80, 627.80, 655.70],
    "eff": [0.0, 79.50, 73.87, 60.70, 75.95, 76.21, 82.97, 68.24, 65.57],
    "pf": [0.105, 0.171, 0.251, 0.321, 0.323, 0.325, 0.317, 0.396, 0.410],
    "slip": [0.00, 0.60, 1.00, 1.33, 1.53, 1.87, 2.20, 2.40, 2.60],
    "R_drum": 0.097
}

def solve_equivalent_circuit(V_line, f, slip, R1, X1, Rc, Xm, R2_prime, X2_prime, P_rot, poles=4):
    V_ph = V_line / np.sqrt(3)
    Ns = 120 * f / poles
    omega_s = 2 * np.pi * Ns / 60
    omega_m = (1.0 - slip) * omega_s
    
    Z1 = R1 + 1j * X1
    Ym = 1.0/Rc + 1.0/(1j * Xm)
    Zm = 1.0/Ym
    
    if slip <= 0:
        Zp = Zm
    else:
        Z2 = R2_prime / slip + 1j * X2_prime
        Zp = (Zm * Z2) / (Zm + Z2)
        
    Zin = Z1 + Zp
    I1 = V_ph / Zin
    I1_abs = np.abs(I1)
    
    P_in = 3.0 * np.real(V_ph * np.conj(I1))
    E1 = V_ph - I1 * Z1
    E1_abs = np.abs(E1)
    
    if slip <= 0:
        I2_prime = 0.0
        P_ag = 0.0
    else:
        I2_prime = E1 / Z2
        P_ag = 3.0 * np.abs(I2_prime)**2 * R2_prime / slip
        
    I2_prime_abs = np.abs(I2_prime)
    P_s_cu = 3.0 * I1_abs**2 * R1
    P_core = 3.0 * E1_abs**2 / Rc
    P_r_cu = slip * P_ag
    P_conv = (1.0 - slip) * P_ag
    
    P_out = P_conv - P_rot
    if P_out < 0:
        P_out = 0.0
        
    T_dev = P_ag / omega_s
    T_shaft = P_out / omega_m if omega_m > 0 else 0.0
    eff = (P_out / P_in * 100.0) if P_in > 0 else 0.0
    pf = np.real(I1) / I1_abs if I1_abs > 0 else 1.0
    
    return {
        "I1": float(I1_abs),
        "P_in": float(P_in),
        "I2_prime": float(I2_prime_abs),
        "P_s_cu": float(P_s_cu),
        "P_core": float(P_core),
        "P_r_cu": float(P_r_cu),
        "P_conv": float(P_conv),
        "P_out": float(P_out),
        "T_dev": float(T_dev),
        "T_shaft": float(T_shaft),
        "eff": float(eff),
        "pf": float(pf)
    }

def fit_motor_parameters(V_line_arr, I_line_arr, P_in_arr, Speed_arr, Torque_arr, f=50, poles=4):
    Ns = 120 * f / poles
    slips = (Ns - np.array(Speed_arr)) / Ns
    
    def loss_func(params):
        R1, R2_prime, X1, X2_prime, Xm, Rc, P_rot = params
        error = 0.0
        for i in range(len(Speed_arr)):
            res = solve_equivalent_circuit(V_line_arr[i], f, slips[i], R1, X1, Rc, Xm, R2_prime, X2_prime, P_rot, poles)
            error += ( (res["I1"] - I_line_arr[i]) / I_line_arr[i] )**2
            error += ( (res["P_in"] - P_in_arr[i]) / P_in_arr[i] )**2
            if Torque_arr[i] > 0:
                error += ( (res["T_shaft"] - Torque_arr[i]) / Torque_arr[i] )**2
        return error
        
    bounds = [
        (0.5, 4.0),     # R1
        (0.5, 4.0),     # R2_prime
        (1.0, 10.0),    # X1
        (1.0, 10.0),    # X2_prime
        (20.0, 150.0),  # Xm
        (100.0, 1500.0),# Rc
        (10.0, 300.0)   # P_rot
    ]
    x0 = [1.5, 1.8, 3.5, 3.5, 80.0, 500.0, 100.0]
    res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
    return res.x

# Fit parameters for both datasets
params_415v = fit_motor_parameters(data_415v["V_line"], data_415v["I_line"], data_415v["P_in"], data_415v["Speed"], data_415v["T_shaft"])
params_220v = fit_motor_parameters(data_220v["V_line"], data_220v["I_line"], data_220v["P_in"], data_220v["Speed"], data_220v["T_shaft"])

def generate_motor_synthetic_data(params, nominal_voltage, num_points=350):
    R1, R2_prime, X1, X2_prime, Xm, Rc, P_rot = params
    np.random.seed(42)
    V_arr = np.random.uniform(nominal_voltage * 0.8, nominal_voltage * 1.15, num_points)
    slip_arr = np.random.uniform(0.0, 0.08, num_points)
    f_arr = np.random.uniform(45.0, 55.0, num_points)
    
    V_list, slip_list, f_list = [], [], []
    I1_list, P_in_list, T_list, P_out_list, eff_list, pf_list = [], [], [], [], [], []
    
    for i in range(num_points):
        v = V_arr[i]
        s = slip_arr[i]
        freq = f_arr[i]
        res = solve_equivalent_circuit(v, freq, s, R1, X1, Rc, Xm, R2_prime, X2_prime, P_rot)
        V_list.append(v)
        slip_list.append(s)
        f_list.append(freq)
        I1_list.append(res["I1"])
        P_in_list.append(res["P_in"])
        T_list.append(res["T_shaft"])
        P_out_list.append(res["P_out"])
        eff_list.append(res["eff"])
        pf_list.append(res["pf"])
        
    return pd.DataFrame({
        "Voltage": V_list,
        "Slip": slip_list,
        "Frequency": f_list,
        "I1": I1_list,
        "P_in": P_in_list,
        "Torque": T_list,
        "P_out": P_out_list,
        "Efficiency": eff_list,
        "PowerFactor": pf_list
    })

df_synth_415v = generate_motor_synthetic_data(params_415v, 415.0)
df_synth_220v = generate_motor_synthetic_data(params_220v, 220.0)

# Train Motor Models
models_415v = train_and_serialize_models(
    df_synth_415v, 
    ["Voltage", "Slip", "Frequency"], 
    ["Efficiency", "PowerFactor", "Torque", "I1", "P_in"]
)
models_220v = train_and_serialize_models(
    df_synth_220v, 
    ["Voltage", "Slip", "Frequency"], 
    ["Efficiency", "PowerFactor", "Torque", "I1", "P_in"]
)

# -------------------------------------------------------------
# PART 2: ALTERNATOR MODEL PIPELINE
# -------------------------------------------------------------

# Experimental 3-Phase Synchronous Alternator Load Test Data
data_alternator = {
    "V_line": [415.0, 410.0, 400.0, 390.0, 375.0, 360.0, 350.0, 340.0, 325.0, 310.0, 290.0, 270.0],
    "I_line": [0.0, 1.0, 2.0, 2.8, 3.4, 4.2, 4.3, 4.8, 5.3, 5.8, 6.34, 6.8],
    "I_f": [1.1, 1.1, 1.06, 1.05, 1.05, 1.05, 1.05, 1.04, 1.04, 1.041, 1.04, 1.04],
    "P_out": [0.0, 710.14, 1385.64, 1891.39, 2208.36, 2618.86, 2606.73, 2826.70, 2983.45, 3114.22, 3184.45, 3180.04]
}

# Fit parameters: V_ph^2 = E_ph^2 - 2 * Ra * (V_ph * Ia) - (Ra^2 + Xs^2) * Ia^2
V_ph_exp = np.array(data_alternator["V_line"], dtype=float) / np.sqrt(3)
I_a_exp = np.array(data_alternator["I_line"], dtype=float)
y_a = V_ph_exp**2
x1_a = -2 * V_ph_exp * I_a_exp
x2_a = -(I_a_exp**2)
X_design_a = np.column_stack((np.ones(len(y_a)), x1_a, x2_a))

beta_a, residuals_a, rank_a, s_a = np.linalg.lstsq(X_design_a, y_a, rcond=None)
E_ph_est = float(np.sqrt(beta_a[0]))
R_a_est = float(beta_a[1])
X_s_est = float(np.sqrt(beta_a[2] - R_a_est**2) if beta_a[2] > R_a_est**2 else 24.64)
r2_fit_a = float(1 - (np.sum((y_a - (X_design_a @ beta_a))**2) / np.sum((y_a - np.mean(y_a))**2)))

def generate_alternator_synthetic_data(E_ph, R_a, X_s, num_points=350):
    np.random.seed(42)
    If_arr = np.random.uniform(0.8, 1.3, num_points)
    Ia_arr = np.random.uniform(0.0, 7.0, num_points)
    phi_arr = np.random.uniform(-0.6, 0.6, num_points) # lagging (-) to leading (+)
    
    If_list, Ia_list, phi_list = [], [], []
    VR_list, eff_list, loss_list, Vt_line_list, Eph_list = [], [], [], [], []
    
    for i in range(num_points):
        iff = If_arr[i]
        ia = Ia_arr[i]
        phi = phi_arr[i]
        
        e_ph = E_ph * (iff / 1.05)
        
        B = 2 * ia * (R_a * math.cos(phi) + X_s * math.sin(phi))
        C = ia**2 * (R_a**2 + X_s**2) - e_ph**2
        disc = B**2 - 4*C
        
        if disc >= 0:
            vt_ph = max(20.0, (-B + math.sqrt(disc)) / 2)
        else:
            vt_ph = 20.0
            
        vt_line = vt_ph * math.sqrt(3)
        vr = ((e_ph - vt_ph) / vt_ph) * 100.0 if vt_ph > 0 else 100.0
        vr = min(100.0, max(-20.0, vr))
        
        p_out = 3.0 * vt_ph * ia * math.cos(phi)
        p_cu = 3.0 * (ia**2) * R_a
        p_loss = p_cu + 250.0
        eff = (p_out / (p_out + p_loss) * 100.0) if (p_out + p_loss) > 0 else 0.0
        eff = min(100.0, max(0.0, eff))
        
        If_list.append(iff)
        Ia_list.append(ia)
        phi_list.append(phi)
        VR_list.append(vr)
        eff_list.append(eff)
        loss_list.append(p_loss)
        Vt_line_list.append(vt_line)
        Eph_list.append(e_ph)
        
    return pd.DataFrame({
        "FieldCurrent": If_list,
        "ArmatureCurrent": Ia_list,
        "PFAngle": phi_list,
        "VoltageRegulation": VR_list,
        "Efficiency": eff_list,
        "Losses": loss_list,
        "Vt_line": Vt_line_list,
        "Eph": Eph_list
    })

df_synth_alt = generate_alternator_synthetic_data(E_ph_est, R_a_est, X_s_est)

# Train Alternator Models
models_alternator = train_and_serialize_models(
    df_synth_alt,
    ["FieldCurrent", "ArmatureCurrent", "PFAngle"],
    ["VoltageRegulation", "Efficiency", "Losses", "Vt_line", "Eph"]
)

# -------------------------------------------------------------
# PART 3: COMPILE CONFIGURATION AND RENDER DASHBOARD
# -------------------------------------------------------------

config_data = {
    "dataset_415v": {
        "experimental": data_415v,
        "fitted_params": {
            "R1": round(params_415v[0], 4),
            "R2_prime": round(params_415v[1], 4),
            "X1": round(params_415v[2], 4),
            "X2_prime": round(params_415v[3], 4),
            "Xm": round(params_415v[4], 4),
            "Rc": round(params_415v[5], 4),
            "P_rot": round(params_415v[6], 4)
        },
        "models": models_415v
    },
    "dataset_220v": {
        "experimental": data_220v,
        "fitted_params": {
            "R1": round(params_220v[0], 4),
            "R2_prime": round(params_220v[1], 4),
            "X1": round(params_220v[2], 4),
            "X2_prime": round(params_220v[3], 4),
            "Xm": round(params_220v[4], 4),
            "Rc": round(params_220v[5], 4),
            "P_rot": round(params_220v[6], 4)
        },
        "models": models_220v
    },
    "alternator": {
        "experimental": data_alternator,
        "fitted_params": {
            "Ra": round(R_a_est, 4),
            "Xs": round(X_s_est, 4),
            "Eph": round(E_ph_est, 2),
            "r2": round(r2_fit_a, 6)
        },
        "models": models_alternator
    }
}

@app.route("/")
def serve_frontend():
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        json_str = json.dumps(config_data)
        injected_html = html_content.replace(
            "const SERVER_DATA = null;",
            f"const SERVER_DATA = {json_str};"
        )
        return render_template_string(injected_html)
    return "Error: index.html not found.", 404

@app.route("/api/record_observation", methods=["POST"])
def record_observation():
    if observations_collection is None:
        return jsonify({"error": "MongoDB not installed or configured."}), 500
    
    data = request.json
    try:
        observations_collection.insert_one(data)
        return jsonify({"status": "success", "message": "Observation saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/get_observations", methods=["GET"])
def get_observations():
    if observations_collection is None:
        return jsonify({"error": "MongoDB not installed or configured."}), 500
        
    facility = request.args.get("facility", None)
    query = {}
    if facility:
        query["facility"] = facility
        
    results = []
    for obs in observations_collection.find(query, {"_id": 0}):
        results.append(obs)
    return jsonify(results), 200

@app.route("/api/login", methods=["POST"])
def login():
    if users_collection is None:
        return jsonify({"error": "Database not configured"}), 500
    
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    user = users_collection.find_one({"username": username, "password": password})
    if user:
        return jsonify({"status": "success", "role": user["role"], "username": user["username"]}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/api/ai/improve", methods=["POST"])
def ai_improve():
    data = request.json
    prompt = data.get("prompt")
    username = data.get("username", "admin")
    
    # 1. Call LLM (Placeholder for Nemotron/OpenAI)
    # We simulate an LLM returning some frontend code changes for demonstration.
    # In a real scenario, you'd pass the file content to the LLM and get a diff/new content.
    llm_generated_code = f"// Improvement generated by AI for prompt: {prompt}\n// Note: This is a simulated LLM response.\n"
    
    # 2. Record improvement intent to database
    if improvements_collection is not None:
        improvements_collection.insert_one({
            "prompt": prompt,
            "username": username,
            "status": "pending",
            "branch_name": f"improvement-{os.urandom(4).hex()}"
        })
        
    # 3. Use GitHub API to create a branch and commit the code
    github_token = os.environ.get("GITHUB_PAT")
    repo_name = "siddardhvanguri-source/digital_machine-s_laboratory-"
    
    if github_token:
        try:
            g = Github(github_token)
            repo = g.get_repo(repo_name)
            
            # Get main branch ref
            main_ref = repo.get_git_ref("heads/main")
            
            # Create new branch
            branch_name = f"improvement-{os.urandom(4).hex()}"
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_ref.object.sha)
            
            # Try to get index.html to update it
            try:
                file_contents = repo.get_contents("index.html", ref=branch_name)
                # For safety in this demo, we just append a comment at the top rather than destroying the file
                new_content = llm_generated_code + file_contents.decoded_content.decode("utf-8")
                repo.update_file(
                    file_contents.path,
                    f"AI Improvement: {prompt}",
                    new_content,
                    file_contents.sha,
                    branch=branch_name
                )
            except Exception as e:
                print("Could not update file on GitHub:", e)
                
            return jsonify({"status": "success", "branch": branch_name, "message": "Improvement branch created!"}), 200
            
        except Exception as e:
            return jsonify({"error": str(e), "message": "GitHub API failed."}), 500
    else:
        return jsonify({"status": "mock_success", "message": "Simulated! No GITHUB_PAT env var found to actually push."}), 200

@app.route("/api/git/branches", methods=["GET"])
def get_git_branches():
    if improvements_collection is not None:
        branches = []
        for imp in improvements_collection.find({}, {"_id": 0}):
            branches.append(imp)
        return jsonify(branches), 200
    return jsonify([]), 200

@app.route("/api/git/publish", methods=["POST"])
def publish_branch():
    data = request.json
    branch_name = data.get("branch_name")
    
    github_token = os.environ.get("GITHUB_PAT")
    repo_name = "siddardhvanguri-source/digital_machine-s_laboratory-"
    
    if github_token and branch_name:
        try:
            g = Github(github_token)
            repo = g.get_repo(repo_name)
            
            # Create a Pull Request and Merge it
            pr = repo.create_pull(
                title=f"Publish AI Improvement: {branch_name}",
                body="Merging AI-generated improvements into main.",
                head=branch_name,
                base="main"
            )
            pr.merge()
            
            # Update DB status
            if improvements_collection is not None:
                improvements_collection.update_one({"branch_name": branch_name}, {"$set": {"status": "published"}})
                
            return jsonify({"status": "success", "message": "Merged and published!"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    # Mock fallback
    if improvements_collection is not None:
        improvements_collection.update_one({"branch_name": branch_name}, {"$set": {"status": "published"}})
    return jsonify({"status": "mock_success", "message": "Simulated publish! Set GITHUB_PAT to actually merge."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
