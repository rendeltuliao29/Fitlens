package com.example.fitlensartest.data

enum class Gender {
    MALE,
    FEMALE,
    OTHER
}

enum class PreferredFit {
    SLIM,
    REGULAR,
    OVERSIZED
}

data class UserInfo(
    // Required fields
    val gender: Gender,
    val age: Int,
    val heightCm: Double,
    val weightKg: Double,

    // Optional fields
    val preferredFit: PreferredFit = PreferredFit.REGULAR,
    val knownChestCircumferenceCm: Double? = null,
    val usualShirtSize: String? = null
)
