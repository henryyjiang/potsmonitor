import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Image(systemName: "heart.fill"); Text("Live") }

            StatsView()
                .tabItem { Image(systemName: "chart.bar.fill"); Text("Stats") }

            ExportView()
                .tabItem { Image(systemName: "square.and.arrow.up"); Text("Export") }

            SettingsView()
                .tabItem { Image(systemName: "gear"); Text("Settings") }
        }
        .tint(.red)
    }
}
