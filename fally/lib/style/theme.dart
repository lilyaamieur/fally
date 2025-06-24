import 'package:flutter/material.dart';

class AppColors {
  static const Color darkGreen = Color(0xFF2E4A26);       // Deep forest green (titles/buttons)
  static const Color mediumGreen = Color(0xFF99b66f);     // Accent green (used in CTAs)
  static const Color lightGreen = Color(0xFFDCEFD3);      // Background / card
  static const Color creamyWhite = Color(0xFFF5F8F1);     // Very soft background
  static const Color textColor = Color(0xFF333333);       // Neutral readable dark text
}

class AppTheme {
  static ThemeData get theme {
    return ThemeData(
      scaffoldBackgroundColor: AppColors.creamyWhite,
      primaryColor: AppColors.darkGreen,
      colorScheme: ColorScheme.light(
        primary: AppColors.darkGreen,
        secondary: AppColors.mediumGreen,
        background: AppColors.creamyWhite,
      ),
      fontFamily: 'Poppins',
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.bold,
          color: AppColors.darkGreen,
        ),
        titleMedium: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w500,
          color: AppColors.textColor,
        ),
        bodyLarge: TextStyle(
          fontSize: 16,
          color: AppColors.textColor,
        ),
        bodyMedium: TextStyle(
          fontSize: 14,
          color: AppColors.textColor,
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.darkGreen,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.mediumGreen,
          foregroundColor: Colors.white,
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
          ),
        ),
      ),
      //cardTheme: const CardTheme(
      //  color: AppColors.lightGreen,
      //  elevation: 2,
      //  shape: RoundedRectangleBorder(
      //    borderRadius: BorderRadius.all(Radius.circular(16)),
      //  ),
      //),
    );
  }
}
