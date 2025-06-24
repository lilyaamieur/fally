import 'package:flutter/material.dart';
import 'pages/measure/ar_measure.dart';
import 'pages/home/home.dart';
import 'package:fally/style/theme.dart';
void main() {
  runApp(FallyApp());
}

class FallyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fally - El Hamma',
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFF1F6EF),
        fontFamily: 'QuickSand',
        primaryColor: const Color(0xFF386641),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF386641),
          background: const Color(0xFFF1F6EF),
          primary: const Color(0xFF386641),
          secondary: const Color(0xFF6A994E),
        ),
        textTheme: const TextTheme(
          titleLarge: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Color(0xFF1B4332),
          ),
          bodyMedium: TextStyle(fontSize: 16, color: Color(0xFF2D6A4F)),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF6A994E),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Stack(
        children: [
          // 🌳 Background Image (Full screen, covers entire layout)
          Positioned.fill(
            child: Image.asset(
              'assets/images/forest_illustration.jpg',
              fit: BoxFit.cover,
            ),
          ),

          // 🌫️ Black transparent overlay (adjust opacity here)
          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(0.5), // 0.3 to 0.6 works well
            ),
          ),

          // 📋 Foreground content (text + button)
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: 24.0,
                vertical: 32,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start,
                spacing: 250,
                children: [
                  // Title + Subtitle
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        child: Image.asset(
                          'assets/images/logo.png',
                          height: 150,
                        ),
                      ),
                      const SizedBox(height: 50),
                      Text(
                        "Welcome to",
                        style: theme.textTheme.titleLarge!.copyWith(
                          color: Colors.white,
                          fontSize: 40,
                          shadows: [
                            const Shadow(
                              blurRadius: 3,
                              color: Colors.black38,
                              offset: Offset(1, 1),
                            ),
                          ],
                        ),
                      ),
                      Text(
                        "Fally",
                        style: theme.textTheme.titleLarge!.copyWith(
                          color: Colors.white,
                          fontSize: 40,
                          shadows: [
                            Shadow(
                              blurRadius: 3,
                              color: Colors.black38,
                              offset: Offset(1, 1),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 30),
                      Text(
                        "Help us protect El Hamma Park by detecting falling branches using AI & AR tools.",
                        style: theme.textTheme.bodyMedium!.copyWith(
                          color: Colors.white70,
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          shadows: [
                            Shadow(
                              blurRadius: 2,
                              color: Colors.black26,
                              offset: Offset(1, 1),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),

                  // AR Detection Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => HomePageWithNav(),
                          ),
                        );
                      },
                      icon: const Icon(Icons.nature_people),
                      label: const Text(
                        'Open AR Detection',
                        style: TextStyle(
                          color: Colors.white, // Set font color here
                          fontFamily:
                              'Quicksand', // Optional if you're using custom font
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.mediumGreen,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
