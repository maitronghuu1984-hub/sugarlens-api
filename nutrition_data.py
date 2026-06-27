nutrition_data = {
    "banh_bao": {
        "name": "Bánh bao",
        "carb_per_100g": 35,
        "sugar_per_100g": 6,
        "calories_per_100g": 220,
        "fiber_per_100g": 1.5,
        "advice": "Bánh bao chứa nhiều tinh bột. Người tiểu đường nên ăn khẩu phần nhỏ."
    },

    "banh_bo": {
        "name": "Bánh bò",
        "carb_per_100g": 45,
        "sugar_per_100g": 18,
        "calories_per_100g": 250,
        "fiber_per_100g": 0.8,
        "advice": "Bánh bò thường có nhiều đường, người tiểu đường nên hạn chế."
    },

    "banh_bot_loc": {
        "name": "Bánh bột lọc",
        "carb_per_100g": 38,
        "sugar_per_100g": 2,
        "calories_per_100g": 190,
        "fiber_per_100g": 0.5,
        "advice": "Bánh bột lọc làm từ tinh bột nên cần kiểm soát số lượng."
    },

    "banh_tet": {
        "name": "Bánh tét",
        "carb_per_100g": 40,
        "sugar_per_100g": 1,
        "calories_per_100g": 250,
        "fiber_per_100g": 1.2,
        "advice": "Bánh tét làm từ nếp, chứa nhiều tinh bột. Nên ăn ít."
    },

    "banh_beo": {
        "name": "Bánh bèo",
        "carb_per_100g": 30,
        "sugar_per_100g": 1,
        "calories_per_100g": 160,
        "fiber_per_100g": 0.7,
        "advice": "Bánh bèo có tinh bột từ bột gạo. Nên ăn khẩu phần nhỏ."
    }
}


def diabetes_risk(carb_g, sugar_g):
    if carb_g >= 45 or sugar_g >= 15:
        return (
            "Cao",
            "Khẩu phần này có lượng carbohydrate hoặc đường khá cao. Nên hạn chế hoặc giảm khẩu phần."
        )

    elif carb_g >= 25 or sugar_g >= 8:
        return (
            "Trung bình",
            "Có thể dùng với khẩu phần nhỏ, nên ăn kèm rau và theo dõi đường huyết."
        )

    else:
        return (
            "Thấp",
            "Khẩu phần tương đối phù hợp, tuy nhiên vẫn cần theo dõi theo tình trạng sức khỏe cá nhân."
        )


def calculate_nutrition(food_key, portion_g):
    if food_key not in nutrition_data:
        return None

    food = nutrition_data[food_key]

    carb = portion_g * food["carb_per_100g"] / 100
    sugar = portion_g * food["sugar_per_100g"] / 100
    calories = portion_g * food["calories_per_100g"] / 100
    fiber = portion_g * food["fiber_per_100g"] / 100

    risk_level, risk_advice = diabetes_risk(carb, sugar)

    return {
        "food_key": food_key,
        "food_name": food["name"],
        "portion_g": portion_g,
        "carbohydrate_g": round(carb, 1),
        "sugar_g": round(sugar, 1),
        "calories": round(calories, 1),
        "fiber_g": round(fiber, 1),
        "risk_level": risk_level,
        "risk_advice": risk_advice,
        "food_advice": food["advice"]
    }