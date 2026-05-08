import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Image(systemName: "heart.fill"); Text("Live") }

            StatsView()
                .tabItem { Image(systemName: "chart.bar.fill"); Text("Stats") }

            SettingsView()
                .tabItem { Image(systemName: "gear"); Text("Settings") }
        }
        .tint(.red)
    }
}
