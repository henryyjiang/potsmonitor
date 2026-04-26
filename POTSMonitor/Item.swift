//
//  Item.swift
//  POTSMonitor
//
//  Created by Henry on 4/25/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
